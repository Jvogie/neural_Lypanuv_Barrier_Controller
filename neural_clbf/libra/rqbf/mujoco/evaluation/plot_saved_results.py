import pandas as pd
import plotly.graph_objects as go
import argparse
import logging
import numpy as np
from typing import Union

def plot_bf_contour(results_df_bf: Union[str, pd.DataFrame]) -> None:
    """Plot Barrier Function contour from CSV data or DataFrame."""
    try:
        ## Load data if string path provided (otherwise use directly provided DataFrame)
        if isinstance(results_df_bf, str):
            results_df_bf = pd.read_csv(results_df_bf)
        
        def extract_barrier_value(x):
            try:
                nums = [float(n) for n in str(x).strip('[]').split()]
                return float(nums[2])  # Taking the third value as the other experiments do
            except:
                return None
        
        for col in ['$x$', '$y$', '$z$']:
            results_df_bf[col] = pd.to_numeric(results_df_bf[col])
        
        results_df_bf['Barrier Function'] = results_df_bf['Barrier Function'].apply(extract_barrier_value)
        results_df_bf = results_df_bf.dropna()
        
        ## Filter out points where barrier function is small
        threshold = 60.0 ## change to allow for "less risky" regions
        results_df_bf = results_df_bf[abs(results_df_bf['Barrier Function']) > threshold]
        
        if len(results_df_bf) == 0:
            logging.error("No data points remain after filtering")
            return
        
        x_vals = sorted(results_df_bf['$x$'].unique())
        y_vals = sorted(results_df_bf['$y$'].unique())
        z_vals = sorted(results_df_bf['$z$'].unique())
        
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
        barrier_values = np.zeros_like(X)
        
        for idx, row in results_df_bf.iterrows():
            i = x_vals.index(row['$x$'])
            j = y_vals.index(row['$y$'])
            k = z_vals.index(row['$z$'])
            barrier_values[i,j,k] = row['Barrier Function']
        
        fig = go.Figure(data=[go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=barrier_values.flatten(),
            isomin=threshold,
            isomax=barrier_values.max(),
            surface_count=5,  ## number of isosurfaces, 5 works well
            colorscale='RdYlBu',
            opacity=0.7,
            caps=dict(x_show=False, y_show=False, z_show=False),  ## hide caps
            colorbar=dict(
                title='Barrier Function Value',
                titleside='right'
            )
        )])
        
        fig.update_layout(
            title=f'Barrier Function Values in 3D Space (|BF| > {threshold})',
            scene=dict(
                xaxis_title='$x$',
                yaxis_title='$y$',
                zaxis_title='$z$',
                camera=dict(
                    eye=dict(x=1.87, y=0.88, z=-0.64)
                ),
                aspectratio=dict(x=1, y=1, z=0.7)
            ),
            width=1000,
            height=800,
            showlegend=False
        )
        
        fig.write_html("bf_contour_3d.html")
        fig.show()
        logging.info("Successfully plotted BF values in 3D")
        
    except Exception as e:
        logging.error(f"Error plotting BF values: {str(e)}")
        raise

def plot_clf_contour(results_df_clf: Union[str, pd.DataFrame]) -> None:
    """Plot Control Lyapunov Function values in 3D."""
    try:
        ## Load data if string path provided (otherwise use directly provided DataFrame)
        if isinstance(results_df_clf, str):
            results_df_clf = pd.read_csv(results_df_clf)
        
        for col in ['$x$', '$y$', '$z$']:
            if col in results_df_clf.columns:
                results_df_clf[col] = pd.to_numeric(results_df_clf[col], errors='coerce')
        
        if 'V' in results_df_clf.columns:
            if isinstance(results_df_clf['V'].iloc[0], str):
                # If V is stored as string, extract the first number
                results_df_clf['V'] = results_df_clf['V'].apply(
                    lambda x: float(str(x).strip('[]').split()[0]) if pd.notnull(x) else None
                )
            else:
                # If V is already numeric, just convert to float
                results_df_clf['V'] = pd.to_numeric(results_df_clf['V'], errors='coerce')
        
        results_df_clf = results_df_clf.dropna()
        
        x_vals = sorted(results_df_clf['$x$'].unique())
        y_vals = sorted(results_df_clf['$y$'].unique())
        z_vals = sorted(results_df_clf['$z$'].unique())
        
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
        clf_values = np.zeros_like(X)

        for idx, row in results_df_clf.iterrows():
            try:
                i = x_vals.index(row['$x$'])
                j = y_vals.index(row['$y$'])
                k = z_vals.index(row['$z$'])
                clf_values[i,j,k] = row['V']
            except (ValueError, IndexError) as e:
                logging.warning(f"Skipping point due to indexing error: {e}")
                continue
        
        fig = go.Figure(data=[go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=clf_values.flatten(),
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
                xaxis_title='$x$',
                yaxis_title='$y$',
                zaxis_title='$z$',
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
        fig.show()
        logging.info("Successfully plotted CLF values in 3D")
        
    except Exception as e:
        logging.error(f"Error plotting CLF values: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Plot BF and/or CLF contours from CSV data')
    parser.add_argument('--bf', type=str, help='Path to BF contour CSV file')
    parser.add_argument('--clf', type=str, help='Path to CLF contour CSV file')
    args = parser.parse_args()
    
    if not args.bf and not args.clf:
        parser.error("At least one of --bf or --clf must be specified")
    
    if args.bf:
        logging.info(f"Plotting BF contour from {args.bf}")
        plot_bf_contour(args.bf)

    if args.clf:
        logging.info(f"Plotting CLF contour from {args.clf}")
        plot_clf_contour(args.clf)