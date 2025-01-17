"""Plot a CLF contour"""
from typing import cast, List, Tuple, Optional, TYPE_CHECKING

import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
import numpy as np
import plotly.graph_objects as go

from neural_clbf.experiments import Experiment
from neural_clbf.controllers.neural_clbf_controller import NeuralCLBFController

if TYPE_CHECKING:
    from neural_clbf.controllers import Controller, CLFController  # noqa


def process_point_clf(args):
    """Process a single point for CLF computation."""
    i, j, k, x_val, y_val, z_val, default_state, controller, indices, labels = args
    x = (
        default_state.clone()
        .detach()
        .reshape(1, controller.dynamics_model.n_dims)
    )
    x[0, indices[0]] = x_val
    x[0, indices[1]] = y_val
    x[0, indices[2]] = z_val

    V = controller.V(x)
    is_goal = controller.dynamics_model.goal_mask(x).all()
    is_safe = controller.dynamics_model.safe_mask(x).all()
    is_unsafe = controller.dynamics_model.unsafe_mask(x).all()

    _, r = controller.solve_CLF_QP(x)
    relaxation = r.max()

    P = controller.dynamics_model.P.type_as(x)
    x0 = controller.dynamics_model.goal_point.type_as(x)
    P = P.reshape(
        1,
        controller.dynamics_model.n_dims,
        controller.dynamics_model.n_dims,
    )
    V_nominal = 0.5 * F.bilinear(x - x0, x - x0, P).squeeze()

    return {
        labels[0]: x_val.cpu().numpy().item(),
        labels[1]: y_val.cpu().numpy().item(),
        labels[2]: z_val.cpu().numpy().item(),
        "V": V.detach().cpu().numpy().item(),
        "QP relaxation": relaxation.detach().cpu().numpy().item(),
        "Goal region": is_goal.cpu().numpy().item(),
        "Safe region": is_safe.cpu().numpy().item(),
        "Unsafe region": is_unsafe.cpu().numpy().item(),
        "Linearized V": V_nominal.detach().cpu().numpy().item(),
    }


class CLFContourExperiment(Experiment):
    """An experiment for plotting the contours of learned CLFs in 3D"""

    def __init__(
        self,
        name: str,
        domain: Optional[List[Tuple[float, float]]] = None,
        n_grid: int = 20,
        batch_size: int = 100,
        x_axis_index: int = 0,
        y_axis_index: int = 1,
        z_axis_index: int = 2,
        x_axis_label: str = "$x$",
        y_axis_label: str = "$y$",
        z_axis_label: str = "$z$",
        default_state: Optional[torch.Tensor] = None,
    ):
        """Initialize an experiment for plotting the value of the CLF over selected
        state dimensions.

        args:
            name: the name of this experiment
            domain: a list of three tuples specifying the plotting range,
                    one for each state dimension.
            n_grid: the number of points in each direction at which to compute V
            x_axis_index: the index of the state variable to plot on the x axis
            y_axis_index: the index of the state variable to plot on the y axis
            z_axis_index: the index of the state variable to plot on the z axis
            x_axis_label: the label for the x axis
            y_axis_label: the label for the y axis
            z_axis_label: the label for the z axis
            default_state: 1 x dynamics_model.n_dims tensor of default state values
        """
        super(CLFContourExperiment, self).__init__(name)

        if domain is None:
            domain = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
        elif len(domain) < 3:
            # If domain is provided but has fewer than 3 elements, extend it
            while len(domain) < 3:
                domain.append((-1.0, 1.0))
        
        self.domain = domain
        self.n_grid = n_grid
        self.batch_size = batch_size
        self.x_axis_index = x_axis_index
        self.y_axis_index = y_axis_index
        self.z_axis_index = z_axis_index
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.z_axis_label = z_axis_label
        self.default_state = default_state

    @torch.no_grad()
    def run(self, controller_under_test: "NeuralCLBFController") -> pd.DataFrame:
        """Run the experiment and save results to CSV."""
        device = "cpu"
        if hasattr(controller_under_test, "device"):
            device = controller_under_test.device

        x = np.linspace(self.domain[0][0], self.domain[0][1], self.n_grid)
        y = np.linspace(self.domain[1][0], self.domain[1][1], self.n_grid)
        z = np.linspace(self.domain[2][0], self.domain[2][1], self.n_grid)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        
        n_points = len(points)
        results = []
        
        for i in tqdm.trange(0, n_points, self.batch_size, desc="Computing CLF"):
            batch_points = points_tensor[i:i+self.batch_size]
            
            x = torch.zeros((len(batch_points), controller_under_test.dynamics_model.n_dims), 
                          device=device)
            x[:, self.x_axis_index] = batch_points[:, 0]
            x[:, self.y_axis_index] = batch_points[:, 1]
            x[:, self.z_axis_index] = batch_points[:, 2]
            
            V = controller_under_test.V(x)
            
            for j in range(len(batch_points)):
                results.append({
                    self.x_axis_label: batch_points[j, 0].cpu().numpy(),
                    self.y_axis_label: batch_points[j, 1].cpu().numpy(),
                    self.z_axis_label: batch_points[j, 2].cpu().numpy(),
                    "V": V[j].cpu().numpy()
                })

        results_df = pd.DataFrame(results)
        results_df.to_csv("clf_contour_data.csv", index=False)
        
        return results_df

    def plot(
        self,
        controller_under_test: "Controller",
        results_df: pd.DataFrame,
        display_plots: bool = False,
    ) -> List[Tuple[str, go.Figure]]:
        """Plot using Plotly."""
        x_vals = sorted(results_df[self.x_axis_label].unique())
        y_vals = sorted(results_df[self.y_axis_label].unique())
        z_vals = sorted(results_df[self.z_axis_label].unique())
        
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
        clf_values = np.zeros_like(X)
        
        for idx, row in results_df.iterrows():
            i = x_vals.index(row[self.x_axis_label])
            j = y_vals.index(row[self.y_axis_label])
            k = z_vals.index(row[self.z_axis_label])
            clf_values[i,j,k] = row['V']
        
        fig = go.Figure(data=[go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=clf_values.flatten(),
            isomin=clf_values.min(),
            isomax=clf_values.max(),
            surface_count=10,
            colorscale='Viridis',
            opacity=0.7,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorbar=dict(
                title='CLF Value',
                titleside='right'
            )
        )])
        
        fig.update_layout(
            title='CLF Values in 3D Space',
            scene=dict(
                xaxis_title=self.x_axis_label,
                yaxis_title=self.y_axis_label,
                zaxis_title=self.z_axis_label,
                camera=dict(
                    eye=dict(x=1.87, y=0.88, z=-0.64)
                ),
                aspectratio=dict(x=1, y=1, z=0.7)
            ),
            width=1000,
            height=800,
            showlegend=False
        )
        
        fig.write_html("clf_contour_3d.html")
        
        if display_plots:
            fig.show()
        
        return [("CLF Surface", fig)]


### """Plot a CLF contour"""
# from typing import cast, List, Tuple, Optional, TYPE_CHECKING

# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
# import pandas as pd
# import seaborn as sns
# import torch
# import torch.nn.functional as F
# import tqdm
# import numpy as np
# from skimage import measure
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import matplotlib.colors as mcolors

# from neural_clbf.experiments import Experiment

# if TYPE_CHECKING:
#     from neural_clbf.controllers import Controller, CLFController  # noqa


# def process_point_clf(args):
#     """Process a single point for CLF computation."""
#     i, j, k, x_val, y_val, z_val, default_state, controller, indices, labels = args
#     x = (
#         default_state.clone()
#         .detach()
#         .reshape(1, controller.dynamics_model.n_dims)
#     )
#     x[0, indices[0]] = x_val
#     x[0, indices[1]] = y_val
#     x[0, indices[2]] = z_val

#     V = controller.V(x)
#     is_goal = controller.dynamics_model.goal_mask(x).all()
#     is_safe = controller.dynamics_model.safe_mask(x).all()
#     is_unsafe = controller.dynamics_model.unsafe_mask(x).all()

#     _, r = controller.solve_CLF_QP(x)
#     relaxation = r.max()

#     P = controller.dynamics_model.P.type_as(x)
#     x0 = controller.dynamics_model.goal_point.type_as(x)
#     P = P.reshape(
#         1,
#         controller.dynamics_model.n_dims,
#         controller.dynamics_model.n_dims,
#     )
#     V_nominal = 0.5 * F.bilinear(x - x0, x - x0, P).squeeze()

#     return {
#         labels[0]: x_val.cpu().numpy().item(),
#         labels[1]: y_val.cpu().numpy().item(),
#         labels[2]: z_val.cpu().numpy().item(),
#         "V": V.detach().cpu().numpy().item(),
#         "QP relaxation": relaxation.detach().cpu().numpy().item(),
#         "Goal region": is_goal.cpu().numpy().item(),
#         "Safe region": is_safe.cpu().numpy().item(),
#         "Unsafe region": is_unsafe.cpu().numpy().item(),
#         "Linearized V": V_nominal.detach().cpu().numpy().item(),
#     }


# class CLFContourExperiment(Experiment):
#     """An experiment for plotting the contours of learned CLFs"""

#     def __init__(
#         self,
#         name: str,
#         domain: Optional[List[Tuple[float, float]]] = None,
#         n_grid: int = 20,
#         batch_size: int = 100,
#         x_axis_index: int = 0,
#         y_axis_index: int = 1,
#         z_axis_index: int = 2,
#         x_axis_label: str = "$x$",
#         y_axis_label: str = "$y$",
#         z_axis_label: str = "$z$",
#         default_state: Optional[torch.Tensor] = None,
#         plot_unsafe_region: bool = True,
#     ):
#         """Initialize an experiment for plotting the value of the CLF over selected
#         state dimensions.

#         args:
#             name: the name of this experiment
#             domain: a list of three tuples specifying the plotting range,
#                     one for each state dimension.
#             n_grid: the number of points in each direction at which to compute V
#             x_axis_index: the index of the state variable to plot on the x axis
#             y_axis_index: the index of the state variable to plot on the y axis
#             z_axis_index: the index of the state variable to plot on the z axis
#             x_axis_label: the label for the x axis
#             y_axis_label: the label for the y axis
#             z_axis_label: the label for the z axis
#             default_state: 1 x dynamics_model.n_dims tensor of default state values
#             plot_unsafe_region: True to plot the safe/unsafe region boundaries.
#         """
#         super(CLFContourExperiment, self).__init__(name)

#         if domain is None:
#             domain = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
#         elif len(domain) < 3:
#             while len(domain) < 3:
#                 domain.append((-1.0, 1.0))
        
#         self.domain = domain
#         self.n_grid = n_grid
#         self.batch_size = batch_size
#         self.x_axis_index = x_axis_index
#         self.y_axis_index = y_axis_index
#         self.z_axis_index = z_axis_index
#         self.x_axis_label = x_axis_label
#         self.y_axis_label = y_axis_label
#         self.z_axis_label = z_axis_label
#         self.default_state = default_state
#         self.plot_unsafe_region = plot_unsafe_region

#     @torch.no_grad()
#     def run(self, controller_under_test: "Controller") -> pd.DataFrame:
#         """
#         Run the experiment, likely by evaluating the controller, but the experiment
#         has freedom to call other functions of the controller as necessary (if these
#         functions are not supported by all controllers, then experiments will be
#         responsible for checking compatibility with the provided controller)

#         args:
#             controller_under_test: the controller with which to run the experiment
#         returns:
#             a pandas DataFrame containing the results of the experiment, in tidy data
#             format (i.e. each row should correspond to a single observation from the
#             experiment).
#         """
#         # Sanity check: can only be called on a NeuralCLFController
#         if not (
#             hasattr(controller_under_test, "V")
#             and hasattr(controller_under_test, "solve_CLF_QP")
#         ):
#             raise ValueError("Controller under test must be a CLFController")

#         controller_under_test = cast("CLFController", controller_under_test)

#         # Set up a dataframe to store the results
#         results = []

#         # Set up the plotting grid
#         device = "cpu"
#         if hasattr(controller_under_test, "device"):
#             device = controller_under_test.device  # type: ignore

#         # Create grid points more efficiently
#         x = np.linspace(self.domain[0][0], self.domain[0][1], self.n_grid)
#         y = np.linspace(self.domain[1][0], self.domain[1][1], self.n_grid)
#         z = np.linspace(self.domain[2][0], self.domain[2][1], self.n_grid)
        
#         # Create mesh grid
#         X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
#         points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
#         # Convert to tensor once
#         points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        
#         # Initialize results array
#         n_points = len(points)
#         results = []
        
#         # Process in batches
#         for i in tqdm.trange(0, n_points, self.batch_size, desc="Computing CLF"):
#             batch_points = points_tensor[i:i+self.batch_size]
            
#             # Create state vectors
#             x = torch.zeros((len(batch_points), controller_under_test.dynamics_model.n_dims), 
#                           device=device)
#             x[:, self.x_axis_index] = batch_points[:, 0]
#             x[:, self.y_axis_index] = batch_points[:, 1]
#             x[:, self.z_axis_index] = batch_points[:, 2]
            
#             # Compute all values in batch
#             V = controller_under_test.V(x)
#             is_goal = controller_under_test.dynamics_model.goal_mask(x)
#             is_safe = controller_under_test.dynamics_model.safe_mask(x)
#             is_unsafe = controller_under_test.dynamics_model.unsafe_mask(x)
#             _, r = controller_under_test.solve_CLF_QP(x)
#             relaxation = r.max(dim=1)[0]
            
#             # Compute nominal V
#             P = controller_under_test.dynamics_model.P.type_as(x)
#             x0 = controller_under_test.dynamics_model.goal_point.type_as(x)
#             P = P.reshape(1, controller_under_test.dynamics_model.n_dims, controller_under_test.dynamics_model.n_dims)
#             V_nominal = 0.5 * torch.bmm(torch.bmm((x - x0), P), (x - x0).transpose(1, 2)).squeeze()
            
#             # Store results
#             for j in range(len(batch_points)):
#                 results.append({
#                     self.x_axis_label: batch_points[j, 0].cpu().numpy(),
#                     self.y_axis_label: batch_points[j, 1].cpu().numpy(),
#                     self.z_axis_label: batch_points[j, 2].cpu().numpy(),
#                     "V": V[j].cpu().numpy(),
#                     "QP relaxation": relaxation[j].cpu().numpy(),
#                     "Goal region": is_goal[j].cpu().numpy(),
#                     "Safe region": is_safe[j].cpu().numpy(),
#                     "Unsafe region": is_unsafe[j].cpu().numpy(),
#                     "Linearized V": V_nominal[j].cpu().numpy()
#                 })

#         return pd.DataFrame(results)

#     def plot(
#         self,
#         controller_under_test: "Controller",
#         results_df: pd.DataFrame,
#         display_plots: bool = False,
#     ) -> List[Tuple[str, figure]]:
#         """Plot the results in 3D."""
#         sns.set_theme(context="talk", style="white")

#         fig = plt.figure(figsize=(15, 10))
#         ax = fig.add_subplot(111, projection='3d')

#         # Create a grid for the 3D plot
#         x = np.array(results_df[self.x_axis_label].unique())
#         y = np.array(results_df[self.y_axis_label].unique())
#         z = np.array(results_df[self.z_axis_label].unique())
        
#         # Convert CLF values to 3D array
#         V = np.array(results_df["V"].values).reshape(self.n_grid, self.n_grid, self.n_grid)

#         # Create isosurfaces at different levels
#         levels = np.linspace(V.min(), V.max(), 10)
#         cmap = plt.cm.viridis
#         norm = mcolors.Normalize(vmin=V.min(), vmax=V.max())
        
#         # Plot isosurfaces for CLF
#         for level in levels:
#             try:
#                 verts, faces, _, _ = measure.marching_cubes(V, level)
                
#                 # Scale vertices to match actual coordinates
#                 verts[:, 0] = x[0] + (x[-1] - x[0]) * verts[:, 0] / (self.n_grid - 1)
#                 verts[:, 1] = y[0] + (y[-1] - y[0]) * verts[:, 1] / (self.n_grid - 1)
#                 verts[:, 2] = z[0] + (z[-1] - z[0]) * verts[:, 2] / (self.n_grid - 1)
                
#                 mesh = Poly3DCollection(verts[faces])
#                 mesh.set_alpha(0.3)
#                 color = cmap(norm(level))
#                 mesh.set_facecolor(color)
#                 ax.add_collection3d(mesh)
#             except:
#                 continue

#         # Plot safe/unsafe regions if specified
#         if self.plot_unsafe_region:
#             safe_region = np.array(results_df["Safe region"].values).reshape(self.n_grid, self.n_grid, self.n_grid)
#             unsafe_region = np.array(results_df["Unsafe region"].values).reshape(self.n_grid, self.n_grid, self.n_grid)
            
#             try:
#                 # Plot safe region boundary
#                 verts, faces, _, _ = measure.marching_cubes(safe_region, 0.5)
#                 verts[:, 0] = x[0] + (x[-1] - x[0]) * verts[:, 0] / (self.n_grid - 1)
#                 verts[:, 1] = y[0] + (y[-1] - y[0]) * verts[:, 1] / (self.n_grid - 1)
#                 verts[:, 2] = z[0] + (z[-1] - z[0]) * verts[:, 2] / (self.n_grid - 1)
#                 mesh = Poly3DCollection(verts[faces])
#                 mesh.set_alpha(0.2)
#                 mesh.set_facecolor('green')
#                 ax.add_collection3d(mesh)
                
#                 # Plot unsafe region boundary
#                 verts, faces, _, _ = measure.marching_cubes(unsafe_region, 0.5)
#                 verts[:, 0] = x[0] + (x[-1] - x[0]) * verts[:, 0] / (self.n_grid - 1)
#                 verts[:, 1] = y[0] + (y[-1] - y[0]) * verts[:, 1] / (self.n_grid - 1)
#                 verts[:, 2] = z[0] + (z[-1] - z[0]) * verts[:, 2] / (self.n_grid - 1)
#                 mesh = Poly3DCollection(verts[faces])
#                 mesh.set_alpha(0.2)
#                 mesh.set_facecolor('red')
#                 ax.add_collection3d(mesh)
#             except:
#                 pass

#         # Add colorbar
#         sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#         sm.set_array([])
#         plt.colorbar(sm, ax=ax, label='CLF Value')

#         # Set labels and title
#         ax.set_xlabel(self.x_axis_label)
#         ax.set_ylabel(self.y_axis_label)
#         ax.set_zlabel(self.z_axis_label)
#         ax.set_title("3D CLF Contours")

#         # Set axis limits
#         ax.set_xlim(x.min(), x.max())
#         ax.set_ylim(y.min(), y.max())
#         ax.set_zlim(z.min(), z.max())

#         # Set the view angle
#         ax.view_init(elev=20, azim=45)

#         fig_handle = ("3D CLF Contours", fig)

#         if display_plots:
#             plt.show()
#             return []
#         else:
#             return [fig_handle]
