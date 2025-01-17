from neural_clbf.controllers.neural_clbf_controller import NeuralCLBFController
import torch
import pandas as pd
import tqdm
import numpy as np
from typing import List, Tuple, Optional
from neural_clbf.experiments import Experiment
import plotly.graph_objects as go

def process_point_bf(args):
    """Process a single point for barrier function computation."""
    i, j, k, x_val, y_val, z_val, default_state, controller, indices, labels = args
    x = (
        default_state.clone()
        .detach()
        .reshape(1, controller.dynamics_model.n_dims)
    )
    x[0, indices[0]] = x_val
    x[0, indices[1]] = y_val
    x[0, indices[2]] = z_val

    h_value = controller(x)

    return {
        labels[0]: x_val.cpu().numpy().item(),
        labels[1]: y_val.cpu().numpy().item(),
        labels[2]: z_val.cpu().numpy().item(),
        "Barrier Function": h_value.detach().cpu().numpy().flatten()[0],
    }

class BFContourExperiment(Experiment):
    """An experiment for plotting the contours of learned Barrier Functions"""

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
        super(BFContourExperiment, self).__init__(name)

        ## Make sure domain has 3 elements for 3D plotting
        if domain is None:
            domain = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
        elif len(domain) < 3:
            ## If domain is provided but has fewer than 3 elements, extend it
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
        
        for i in tqdm.trange(0, n_points, self.batch_size, desc="Computing BF"):
            batch_points = points_tensor[i:i+self.batch_size]
            
            x = torch.zeros((len(batch_points), controller_under_test.dynamics_model.n_dims), 
                          device=device)
            x[:, self.x_axis_index] = batch_points[:, 0]
            x[:, self.y_axis_index] = batch_points[:, 1]
            x[:, self.z_axis_index] = batch_points[:, 2]
            
            h_values = controller_under_test(x)
            
            for j in range(len(batch_points)):
                results.append({
                    self.x_axis_label: batch_points[j, 0].cpu().numpy(),
                    self.y_axis_label: batch_points[j, 1].cpu().numpy(),
                    self.z_axis_label: batch_points[j, 2].cpu().numpy(),
                    "Barrier Function": h_values[j].cpu().numpy()
                })

        results_df = pd.DataFrame(results)
        
        results_df.to_csv("bf_contour_data.csv", index=False)
        
        return results_df

    def plot(
        self,
        controller_under_test: "NeuralCLBFController",
        results_df: pd.DataFrame,
        display_plots: bool = False,
    ) -> List[Tuple[str, go.Figure]]:
        """Plot using Plotly."""
        # Handle duplicates by taking the mean
        results_df = results_df.groupby(['$x$', '$y$'])['Barrier Function'].mean().reset_index()
        
        x = sorted(results_df[self.x_axis_label].unique())
        y = sorted(results_df[self.y_axis_label].unique())
        
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for idx, row in results_df.iterrows():
            i = x.index(row[self.x_axis_label])
            j = y.index(row[self.y_axis_label])
            Z[j,i] = row['Barrier Function']
        
        fig = go.Figure(data=[go.Surface(
            x=x,
            y=y,
            z=Z,
            colorscale='Viridis'
        )])
        
        fig.update_layout(
            title='Barrier Function Surface',
            scene=dict(
                xaxis_title=self.x_axis_label,
                yaxis_title=self.y_axis_label,
                zaxis_title='Barrier Function',
            ),
            width=800,
            height=800
        )
        
        fig.write_html("bf_contour_3d.html")
        
        if display_plots:
            fig.show()
        
        return [("Barrier Function Surface", fig)]
    
# from neural_clbf.controllers.neural_clbf_controller import NeuralCLBFController
# import torch
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import tqdm
# import numpy as np
# from typing import List, Tuple, Optional
# from matplotlib.pyplot import figure
# from neural_clbf.experiments import Experiment
# from skimage import measure
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import matplotlib.colors as mcolors

# def process_point_bf(args):
#     """Process a single point for barrier function computation."""
#     i, j, k, x_val, y_val, z_val, default_state, controller, indices, labels = args
#     x = (
#         default_state.clone()
#         .detach()
#         .reshape(1, controller.dynamics_model.n_dims)
#     )
#     x[0, indices[0]] = x_val
#     x[0, indices[1]] = y_val
#     x[0, indices[2]] = z_val

#     h_value = controller(x)

#     return {
#         labels[0]: x_val.cpu().numpy().item(),
#         labels[1]: y_val.cpu().numpy().item(),
#         labels[2]: z_val.cpu().numpy().item(),
#         "Barrier Function": h_value.detach().cpu().numpy().flatten()[0],
#     }

# class BFContourExperiment(Experiment):
#     """An experiment for plotting the contours of learned Barrier Functions"""

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
#     ):
#         super(BFContourExperiment, self).__init__(name)

#         ## Make sure domain has 3 elements for 3D plotting
#         if domain is None:
#             domain = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
#         elif len(domain) < 3:
#             ## If domain is provided but has fewer than 3 elements, extend it
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

#     @torch.no_grad()
#     def run(self, controller_under_test: "NeuralCLBFController") -> pd.DataFrame:
#         device = "cpu"
#         if hasattr(controller_under_test, "device"):
#             device = controller_under_test.device

#         x = np.linspace(self.domain[0][0], self.domain[0][1], self.n_grid)
#         y = np.linspace(self.domain[1][0], self.domain[1][1], self.n_grid)
#         z = np.linspace(self.domain[2][0], self.domain[2][1], self.n_grid)
        
#         X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
#         points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
#         points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        
#         n_points = len(points)
#         results = []
        
#         for i in tqdm.trange(0, n_points, self.batch_size, desc="Computing BF"):
#             batch_points = points_tensor[i:i+self.batch_size]
            
#             ## Create state vectors
#             x = torch.zeros((len(batch_points), controller_under_test.dynamics_model.n_dims), 
#                           device=device)
#             x[:, self.x_axis_index] = batch_points[:, 0]
#             x[:, self.y_axis_index] = batch_points[:, 1]
#             x[:, self.z_axis_index] = batch_points[:, 2]
            
#             ## Compute barrier function values
#             h_values = controller_under_test(x)
            
#             for j in range(len(batch_points)):
#                 results.append({
#                     self.x_axis_label: batch_points[j, 0].cpu().numpy(),
#                     self.y_axis_label: batch_points[j, 1].cpu().numpy(),
#                     self.z_axis_label: batch_points[j, 2].cpu().numpy(),
#                     "Barrier Function": h_values[j].cpu().numpy()
#                 })

#         return pd.DataFrame(results)

#     def plot(
#         self,
#         controller_under_test: "NeuralCLBFController",
#         results_df: pd.DataFrame,
#         display_plots: bool = False,
#     ) -> List[Tuple[str, figure]]:
#         sns.set_theme(context="talk", style="white")

#         fig = plt.figure(figsize=(15, 10))
#         ax = fig.add_subplot(111, projection='3d')

#         x = np.array(results_df[self.x_axis_label].unique())
#         y = np.array(results_df[self.y_axis_label].unique())
#         z = np.array(results_df[self.z_axis_label].unique())
        
#         V = np.array(results_df["Barrier Function"].values).reshape(self.n_grid, self.n_grid, self.n_grid)

#         levels = np.linspace(V.min(), V.max(), 10)
#         cmap = plt.get_cmap('viridis')
#         norm = mcolors.Normalize(vmin=V.min(), vmax=V.max())
        
#         ## Plot isosurfaces
#         for level in levels:
#             try:
#                 verts, faces, _, _ = measure.marching_cubes(V, level)
                
#                 ## Scale vertices to match actual coordinates
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

#         sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#         sm.set_array([])
#         plt.colorbar(sm, ax=ax, label='Barrier Function Value')

#         ax.set_xlabel(self.x_axis_label)
#         ax.set_ylabel(self.y_axis_label)
#         ax.set_zlabel(self.z_axis_label)
#         ax.set_title("3D Barrier Function Contours")

#         ax.set_xlim(x.min(), x.max())
#         ax.set_ylim(y.min(), y.max())
#         ax.set_zlim(z.min(), z.max())

#         ax.view_init(elev=20, azim=45)

#         fig_handle = ("3D Barrier Function Contours", fig)

#         if display_plots:
#             plt.show()
#             return []
#         else:
#             return [fig_handle]
#             # End of Selection