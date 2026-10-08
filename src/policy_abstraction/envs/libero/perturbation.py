"""Manager for handling LIBERO environment perturbations and temporary environment setup."""

import logging
import os

import perturbation  # from LIBERO-PRO, assumes the pythonpath is set up correctly to import it


class PerturbationManager:
    """Handles perturbation configuration and temporary environment creation for LIBERO tasks."""

    def __init__(self, evaluation_cfg):
        """
        Initialize perturbation manager.

        Args:
            evaluation_cfg: Dictionary containing evaluation configuration
        """
        self.cfg = evaluation_cfg

    def setup_perturbed_environment(
        self,
        task_suite_name,
        use_swap,
        use_object,
        use_language,
        use_task,
        use_environment,
    ):
        """
        Setup the perturbed environment based on enabled perturbation flags.

        Args:
            task_suite_name: Name of the task suite
            use_swap: Enable swap perturbation
            use_object: Enable object perturbation
            use_language: Enable language perturbation
            use_task: Enable task perturbation
            use_environment: Enable environment perturbation

        Returns:
            Updated task_suite_name with appropriate suffix
        """
        perturbation_flags = [
            use_swap,
            use_object,
            use_language,
            use_task,
            use_environment,
        ]
        num_active_perturbations = sum(perturbation_flags)

        if num_active_perturbations > 1:
            return self._setup_multi_perturbation_env(
                task_suite_name,
                use_swap,
                use_object,
                use_language,
                use_task,
                use_environment,
            )
        elif num_active_perturbations == 1:
            return self._setup_single_perturbation_env(
                task_suite_name,
                use_swap,
                use_object,
                use_language,
                use_task,
                use_environment,
            )
        else:
            # No perturbations
            return task_suite_name

    def _setup_multi_perturbation_env(
        self,
        task_suite_name,
        use_swap,
        use_object,
        use_language,
        use_task,
        use_environment,
    ):
        """Handle case when multiple perturbations are enabled."""
        bddl_file_path = (
            self.cfg.get("bddl_files_path", "") + task_suite_name + "_temp/"
        )
        init_file_path = self.cfg.get("init_file_dir", "") + task_suite_name + "_temp/"

        # Create log content based on current flags
        log_content = (
            f"{use_swap},{use_object},{use_language},{use_task},{use_environment}"
        )

        should_create_env = self._should_recreate_environment(
            bddl_file_path, init_file_path, log_content
        )

        if should_create_env:
            self._create_temp_environment(bddl_file_path, init_file_path, log_content)

        return task_suite_name + "_temp"

    def _setup_single_perturbation_env(
        self,
        task_suite_name,
        use_swap,
        use_object,
        use_language,
        use_task,
        use_environment,
    ):
        """Handle case when exactly one perturbation is enabled."""
        perturb_key = self._get_active_perturbation_key(
            use_swap, use_object, use_language, use_task, use_environment
        )

        perturbation_suffix = self.cfg.get("perturbation_mapping", {}).get(
            perturb_key, ""
        )
        init_file_path = (
            self.cfg.get("init_file_dir", "")
            + task_suite_name
            + "_"
            + perturbation_suffix
        )

        if not os.path.exists(init_file_path):
            logging.info(f"Creating perturbation environment for {perturb_key}")
            perturbation.create_env(configs=self.cfg)

        return task_suite_name + "_" + perturbation_suffix

    def _get_active_perturbation_key(
        self, use_swap, use_object, use_language, use_task, use_environment
    ):
        """Determine which perturbation is active."""
        if use_swap:
            return "use_swap"
        elif use_object:
            return "use_object"
        elif use_language:
            return "use_language"
        elif use_task:
            return "use_task"
        elif use_environment:
            return "use_environment"
        return "none"

    def _should_recreate_environment(
        self, bddl_file_path, init_file_path, expected_log
    ):
        """Check if environment needs to be recreated based on log file."""
        if not os.path.exists(bddl_file_path) or not os.path.exists(init_file_path):
            return True

        log_file = os.path.join(bddl_file_path, "log.txt")
        if not os.path.exists(log_file):
            return True

        with open(log_file, "r") as f:
            log_contents = f.read().strip()

        return log_contents != expected_log

    def _create_temp_environment(self, bddl_file_path, init_file_path, log_content):
        """Create temporary environment directories and files."""
        # Clean up existing directories if they exist
        self._cleanup_directory(bddl_file_path)
        self._cleanup_directory(init_file_path)

        # Create directories
        os.makedirs(init_file_path, exist_ok=True)
        os.makedirs(bddl_file_path, exist_ok=True)

        # Write log file
        with open(os.path.join(bddl_file_path, "log.txt"), "w") as log_file:
            log_file.write(log_content)

        # Create the environment
        logging.info("Creating perturbed environment with multiple perturbations")
        perturbation.create_env(configs=self.cfg)

    def _cleanup_directory(self, directory):
        """Remove all files and subdirectories in the given directory."""
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
