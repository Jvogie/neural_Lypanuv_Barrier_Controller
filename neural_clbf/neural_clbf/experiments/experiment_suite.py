"""An experiment suite manages a collection of experiments, allowing the user to
run each experiment.
"""
from datetime import datetime
import os
from typing import List, Optional, Tuple, TYPE_CHECKING

from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from pytorch_lightning.loggers import LightningLoggerBase

from .experiment import Experiment

if TYPE_CHECKING:
    from neural_clbf.controllers import Controller  # noqa


matplotlib.use('Agg')


class ExperimentSuite(object):
    """
    A class for managing a collection of experiments.

    This class allows users to run multiple experiments, save results,
    plot data, and log plots for a given controller.

    Attributes:
        experiments (List[Experiment]): A list of Experiment objects comprising the suite.
    """

    def __init__(self, experiments: List[Experiment]):
        """
        Create an ExperimentSuite: an object for managing a collection of experiments.

        args:
            experiments: a list of Experiment objects comprising the suite
        """
        super(ExperimentSuite, self).__init__()
        self.experiments = experiments

    def run_all(self, controller_under_test: "Controller") -> List[pd.DataFrame]:
        """Run all experiments in the suite and return the data from each

        args:
            controller_under_test: the controller with which to run the experiments
        returns:
            a list of DataFrames, one for each experiment
        """
        results = []
        for experiment in self.experiments:
            results.append(experiment.run(controller_under_test))

        return results

    def run_all_and_save_to_csv(
        self, controller_under_test: "Controller", save_dir: str
    ):
        """Run all experiments in the suite and save the results in one directory.

        Results will be saved in a subdirectory save_dir/{timestamp}/...

        args:
            controller_under_test: the controller with which to run the experiments
            save_dir: the path to the directory in which to save the results
        returns:
            a list of DataFrames, one for each experiment
        """
        ## Make sure the given directory exists; create it if it does not
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        ## Get the subdirectory name
        current_datetime = datetime.now()
        timestamp = current_datetime.strftime("%Y-%m-%d_%H_%M_%S")
        subdirectory_path = f"{save_dir}/{timestamp}"

        ## Run and save all experiments (these will create subdirectory if it does not
        ## already exist)
        for experiment in self.experiments:
            experiment.run_and_save_to_csv(controller_under_test, subdirectory_path)

    def run_all_and_plot(
        self, controller_under_test: "Controller", display_plots: bool = False
    ) -> List[Tuple[str, figure]]:
        """
        Run all experiments, plot the results, and return the plot handles. Optionally
        display the plots.

        args:
            controller_under_test: the controller with which to run the experiments
            display_plots: defaults to False. If True, display the plots (blocks until
                           the user responds).
        returns: a list of tuples containing the name of each figure and the figure
                 object.
        """
        ## Create a place to store all the returned handles
        fig_handles = []

        ## Run each experiment, plot, and store the handles
        for experiment in self.experiments:
            fig_handles += experiment.run_and_plot(controller_under_test, display_plots)

        return fig_handles

    def run_all_and_log_plots(
        self,
        controller_under_test: "Controller",
        logger: LightningLoggerBase,
        global_step: int,
    ):
        """
        Run all experiments and log the resulting plots.

        This method runs all experiments in the suite, generates plots,
        and logs them using the provided logger.

        Args:
            controller_under_test (Controller): The controller to be tested in the experiments.
            logger (LightningLoggerBase): The logger used to log the plots.
            global_step (int): The current global step, used for logging.

        Raises:
            ValueError: If the returned fig_handles have an unexpected type.
        """
        fig_handles = self.run_all_and_plot(controller_under_test, display_plots=False)

        ## Check if fig_handles is a single Figure object or a list
        if isinstance(fig_handles, plt.Figure):
            ## If it's a single Figure, convert it to a list with a default name
            fig_handles = [("experiment_plot", fig_handles)]
        elif isinstance(fig_handles, list):
            ## If it's a list, ensure each item is a tuple of (name, Figure)
            fig_handles = [
                (f"experiment_plot_{i}", fig) if isinstance(fig, plt.Figure) else fig
                for i, fig in enumerate(fig_handles)
            ]
        else:
            raise ValueError("Unexpected type for fig_handles")

        for plot_name, figure_handle in fig_handles:
            logger.experiment.add_figure(
                plot_name, figure_handle, global_step=global_step
            )
            plt.close(figure_handle)
