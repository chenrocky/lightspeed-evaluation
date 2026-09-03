"""Evaluation Pipeline - Main evaluation orchestrator."""

import asyncio
import concurrent.futures
import logging
import os
from typing import Optional

import litellm
import tqdm

from lightspeed_evaluation.core.api import APIClient
from lightspeed_evaluation.core.llm.litellm_patch import litellm_state_lock
from lightspeed_evaluation.core.metrics.manager import MetricManager
from lightspeed_evaluation.core.models import (
    EvaluationData,
    EvaluationResult,
    HttpApiAgentConfig,
    SystemConfig,
)
from lightspeed_evaluation.core.output.data_persistence import save_evaluation_data
from lightspeed_evaluation.core.script import ScriptExecutionManager
from lightspeed_evaluation.core.storage import (
    RunInfo,
    BaseStorageBackend,
    create_pipeline_storage_backend,
    get_file_config,
)
from lightspeed_evaluation.core.system import ConfigLoader
from lightspeed_evaluation.core.system.exceptions import StorageError
from lightspeed_evaluation.pipeline.evaluation.amender import APIDataAmender
from lightspeed_evaluation.pipeline.evaluation.errors import EvaluationErrorHandler
from lightspeed_evaluation.pipeline.evaluation.evaluator import MetricsEvaluator
from lightspeed_evaluation.pipeline.evaluation.processor import (
    ConversationProcessor,
    ProcessorComponents,
)

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Evaluation pipeline - orchestrates the evaluation process through different stages.

    Responsibilities:
    - Initialize and coordinate components
    - Orchestrate evaluation flow
    - Collect results
    - Save amended data
    """

    def __init__(self, config_loader: ConfigLoader, output_dir: Optional[str] = None):
        """Initialize evaluation pipeline with config and create components."""
        self.config_loader = config_loader
        if not config_loader.system_config:
            raise ValueError("SystemConfig must be loaded before initializing pipeline")

        self.system_config: SystemConfig = config_loader.system_config
        self.original_data_path: Optional[str] = None
        file_config = get_file_config(config_loader.system_config.storage)
        self.output_dir = output_dir or file_config.output_dir

        self.storage_backend: BaseStorageBackend = create_pipeline_storage_backend(
            config_loader.system_config.storage,
            system_config=config_loader.system_config,
            output_dir_override=output_dir,
        )

        # Initialize components
        self._initialize_components()
        logger.info("Evaluation Pipeline initialized")

    def _initialize_components(self) -> None:
        """Initialize all required components."""
        config = self.config_loader.system_config
        if config is None:
            raise ValueError(
                "SystemConfig must be loaded before initializing components"
            )

        # Metric manager
        metric_manager = MetricManager(config)

        # Create pipeline components
        self.api_client = self._create_api_client()
        api_amender = APIDataAmender(self.api_client)
        error_handler = EvaluationErrorHandler()

        # Create script execution manager
        script_manager = ScriptExecutionManager()

        # Create metrics evaluator with script manager
        metrics_evaluator = MetricsEvaluator(
            self.config_loader, metric_manager, script_manager
        )

        # Create processor components
        processor_components = ProcessorComponents(
            metrics_evaluator=metrics_evaluator,
            api_amender=api_amender,
            error_handler=error_handler,
            metric_manager=metric_manager,
            script_manager=script_manager,
        )

        # Conversation processor
        self.conversation_processor = ConversationProcessor(
            self.config_loader,
            processor_components,
        )

    def _create_api_client(self) -> Optional[APIClient]:
        """Create API client if enabled.

        When the agents layer is active, resolves the default agent config
        so the client uses the correct endpoint_type, api_base, etc.
        Legacy ``api:`` blocks are auto-migrated to ``agents:`` by
        ``SystemConfig.migrate_api_to_agents``, so all configs flow
        through the same path.
        """
        config = self.config_loader.system_config
        if config is None:
            raise ValueError("SystemConfig must be loaded before creating API client")

        if config.agents is None or not config.agents.enabled:
            return None

        _name, agent_dict = config.agents.resolve_agent_config()
        agent_config = HttpApiAgentConfig.model_validate(agent_dict)
        client = APIClient(agent_config)
        logger.info(
            "API client initialized for %s endpoint", agent_config.endpoint_type
        )
        return client

    def run_evaluation(
        self,
        evaluation_data: list[EvaluationData],
        original_data_path: Optional[str] = None,
    ) -> list[EvaluationResult]:
        """Run evaluation on provided data.

        Args:
            evaluation_data: List of conversation data to evaluate
            original_data_path: Path to original data file for saving updates

        Returns:
            List of evaluation results.
        """
        self.original_data_path = original_data_path
        logger.info("Starting evaluation")

        run_name = original_data_path or "evaluation"
        self.storage_backend.initialize(RunInfo(name=run_name))

        try:
            # Process each conversation
            logger.info("Processing conversations")
            results = self._process_eval_data(evaluation_data)
        finally:
            self.storage_backend.set_evaluation_context(evaluation_data)
            self.storage_backend.finalize()
            self.storage_backend.close()

        # Save amended data if API was used
        config = self.config_loader.system_config
        if config is None:
            raise ValueError("SystemConfig must be loaded")
        if config.agents is not None and config.agents.enabled:
            logger.info("Saving amended evaluation data")
            self._save_amended_data(evaluation_data)

        logger.info("Evaluation complete: %d results generated", len(results))
        return results

    CHECKPOINT_FILENAME = ".completed_ids"

    def _checkpoint_path(self) -> str:
        return os.path.join(self.output_dir, self.CHECKPOINT_FILENAME)

    def get_completed_ids(self) -> set[str]:
        """Read completed conversation IDs from checkpoint file."""
        path = self._checkpoint_path()
        if not os.path.exists(path):
            return set()
        with open(path) as f:
            return {line.strip() for line in f if line.strip()}

    def _save_checkpoint(self, conv_id: str) -> None:
        """Append a completed conversation ID to the checkpoint file."""
        with open(self._checkpoint_path(), "a") as f:
            f.write(conv_id + "\n")

    def _process_eval_data(
        self, evaluation_data: list[EvaluationData]
    ) -> list[EvaluationResult]:
        """Process the conversations from the evaluation_data."""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.system_config.core.max_threads
        ) as executor:
            future_to_conv = {
                executor.submit(
                    self.conversation_processor.process_conversation, c
                ): c.conversation_group_id
                for c in evaluation_data
            }
            results: list[EvaluationResult] = []
            failed_conversations = 0
            for future in tqdm.tqdm(
                concurrent.futures.as_completed(future_to_conv),
                total=len(evaluation_data),
            ):
                conv_id = future_to_conv[future]
                try:
                    conversation_results = future.result()
                except Exception:
                    failed_conversations += 1
                    logger.exception(
                        "Conversation %s failed with exception; "
                        "continuing with remaining conversations",
                        conv_id,
                    )
                    continue
                if conversation_results:
                    try:
                        self.storage_backend.save_run(conversation_results)
                    except StorageError as e:
                        logger.warning(
                            "Failed to save results to storage: %s", e
                        )
                results.extend(conversation_results)
                self._save_checkpoint(conv_id)
            if failed_conversations:
                logger.warning(
                    "%d conversation(s) failed during evaluation",
                    failed_conversations,
                )
            return results

    def _save_amended_data(self, evaluation_data: list[EvaluationData]) -> None:
        """Save amended evaluation data with API amendments to output directory."""
        if not self.original_data_path:
            logger.warning("No original data path available, cannot save amended data")
            return

        try:
            amended_file = save_evaluation_data(
                evaluation_data, self.original_data_path, self.output_dir
            )
            if amended_file:
                logger.info("Amended data saved: %s", amended_file)
                logger.info(
                    "To use amended data without new API calls, "
                    "disable the API call using system config & "
                    "replace the original evaluation data file with the amended file"
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Don't fail the evaluation if saving fails
            logger.warning("Failed to save amended data: %s", e)

    def close(self) -> None:
        """Clean up resources.

        Uses a lock to serialize litellm cache teardown across concurrent
        pipelines, since ``litellm.cache`` is process-global state.
        """
        if self.api_client:
            self.api_client.close()

        self.storage_backend.close()

        with litellm_state_lock:
            cache = litellm.cache
            if cache is not None:
                try:
                    # Use getattr to call untyped third-party method
                    disconnect = getattr(cache, "disconnect")
                    asyncio.run(disconnect())
                except (AttributeError, RuntimeError, OSError):
                    logger.debug("litellm cache disconnect raised; ignoring")
                litellm.cache = None
