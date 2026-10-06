
#%%
from tbparse import SummaryReader
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from omegaconf import OmegaConf
from pathlib import Path

from src.localization.config.paths import PathConfig


class LogAnalyzer:
    """
    A class to analyze PyTorch Lightning logs and generate reports and visualizations.
    """
    
    
    
    ERROR_TAGS = ['Error/CT', 'Error/DRMS', 'Error/Euclidean', 'Error/KS', 'Error/TW', 'Error/cep', 'Error/r95']
    ERROR_COLS = ["Euclidean", "TW", "KS", "CT", "DRMS", "cep", "r95"]
    
    def __init__(self, dataset_name,log_dir=None,start_version=None,end_version=None
                 ):
        """
        Initialize the LogAnalyzer.
        
        Args:
            dataset_name (str): Name of the dataset
            csv_dir (str): Directory to save CSV files
            plots_dir (str): Directory to save plots
        """
        path_config = PathConfig()
        self.log_dir = log_dir
        if self.log_dir is None:
            self.log_dir = path_config.lightning_logs
            
        self.dataset_name = dataset_name
        self.reports_dir = path_config.reports/ Path(dataset_name)
        self.csv_dir = self.reports_dir / "csv"
        self.report_path = self.csv_dir / f"report_{self.dataset_name}.csv"
        

        

        self.version_list = [f"version_{x}" for x in range(start_version,end_version)]
        
        self.report_df = None
        
        # Create directories if they don't exist
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    
    def export_logs_to_csv(self):
        """Export TensorBoard logs to CSV files."""
        print(f"Exporting logs for {len(self.version_list)} versions...")
        from tqdm import tqdm
        for version in tqdm(self.version_list):
            try:
                reader = SummaryReader(self.log_dir / version)
                df = reader.scalars
                df.to_csv(self.csv_dir / f"{version}.csv", index=False)
            except Exception as e:
                print(f"Error processing {version}: {e}")
        
        print("Successfully exported logs to CSV")
    
    def generate_report(self):
        """Generate a report dataframe from CSV files and hyperparameters."""
        print(f"Generating report for {len(self.version_list)} versions...")
        
        report_df = []
        
        for version in self.version_list:
            try:
                # Load hyperparameters
                hparams = OmegaConf.load(self.log_dir / version / "hparams.yaml")
                row = dict(hparams)
                
                # Load metrics
                df = pd.read_csv(self.csv_dir / f"{version}.csv")
                row["version"] = version
                
                # Group by tag and extract statistics
                groups = df.groupby("tag")
                
                for tag in self.ERROR_TAGS:
                    try:
                        g = groups.get_group(tag)
                        value = g["value"]
                        
                        row[f"{tag}_min"] = value.min()
                        row[f"{tag}_max"] = value.max()
                        row[f"{tag}_mean"] = value.mean()
                        row[f"{tag}_last"] = value.iloc[-1]
                        row[f"{tag}_100"] = value.iloc[100] if len(value) > 100 else value.iloc[-1]
                    except KeyError:
                        print(f"Warning: Tag '{tag}' not found in {version}")
                        continue
                
                report_df.append(row)
            
            except Exception as e:
                print(f"Error processing {version}: {e}")
        
        self.report_df = pd.DataFrame(report_df)
        self.report_df.to_csv(self.report_path, index=False)
        print(f"Report saved: {self.report_path}")
        
        return self.report_df
    
    def load_report(self):
        """Load existing report from CSV."""
        if not self.report_path.exists():
            print(f"Report file not found: {self.report_path}")
            return None
        self.report_df = pd.read_csv(self.report_path).reset_index(drop=True)
        return self.report_df
    
    def plot_error_analysis(self,post_fix = "last", x_col="coeff_mse_loss", hue_col="coeff_dissimilarity_loss"):
        """
        Create boxplots for error metrics by loss coefficients.
        
        Args:
            x_col (str): Column for x-axis
            hue_col (str): Column for hue grouping
        """
        if self.report_df is None:
            raise ValueError("Report not loaded. Call load_report() or generate_report() first.")
        
        print(f"Generating plots for {len(self.ERROR_COLS)} error metrics...")
        
        for col in self.ERROR_COLS:
            plt.figure(figsize=(10, 10))
            sns.set_theme(style="whitegrid")
            ax = plt.gca()
            
            sns.boxplot(
                data=self.report_df,
                x=x_col,
                y=f"Error/{col}_{post_fix}",
                hue=hue_col,
                width=0.750,
                gap=0.05,
                palette="Set2",
                showfliers=False
            )
            
            ax.legend(title="Time Distance Loss Coefficient")
            ax.set_xlabel("Masked Encoder Loss Coefficient")
            ax.set_ylabel(col)
            ax.set_title(f"Effect of Loss Coefficients on {col} for {self.dataset_name} dataset")
            
            plt.savefig(self.reports_dir / f"{self.dataset_name}_{col}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print("Plots saved successfully")
    
    def create_pivot_table(self, values_col="Error/Euclidean_last", 
                          index_col="coeff_mse_loss", columns_col="coeff_dissimilarity_loss"):
        """
        Create a pivot table and save as LaTeX.
        
        Args:
            values_col (str): Column to pivot
            index_col (str): Index column
            columns_col (str): Columns column
        
        Returns:
            pd.DataFrame: Pivot table
        """
        if self.report_df is None:
            raise ValueError("Report not loaded. Call load_report() or generate_report() first.")
        
        pivot_table = self.report_df.pivot_table(
            index=index_col,
            columns=columns_col,
            values=values_col,
            aggfunc="median"
        )
        
        pivot_table = pivot_table.astype(float).round(2)
        pivot_table = pivot_table.rename_axis(
            index={index_col: "Masked Encoder Loss Coefficient"},
            columns={columns_col: "Time Distance Loss Coefficient"}
        )
        
        pivot_table.to_latex(
            self.reports_dir / f"{self.dataset_name}.tex",
            index=True,
            caption=f"Comparison of loss coefficients on {values_col.split('_')[1]} in {self.dataset_name} dataset",
            label="tab:loss_coefficients",
            float_format="%.2f"
        )
        
        print(f"LaTeX table saved: {self.csv_dir / f'{self.dataset_name}.tex'}")
        return pivot_table
    
    def get_summary_by_model(self, model_name):
        """
        Get summary statistics grouped by model name and loss coefficient.
        
        Args:
            model_name (str): Name of the model
        
        Returns:
            tuple: (summary_list, series_list)
        """
        if self.report_df is None:
            raise ValueError("Report not loaded. Call load_report() or generate_report() first.")
        
        summary_list = []
        series_list = []
        
        filtered_df = self.report_df[self.report_df["model_name"] == model_name].reset_index()
        indices = filtered_df.groupby("coeff_mse_loss").indices
        
        for coeff, idx_list in indices.items():
            metrics = filtered_df.loc[idx_list, [
                "Error/CT_max", "Error/DRMS_min", "Error/Euclidean_min",
                "Error/KS_min", "Error/TW_max"
            ]].mean()
            
            summary_list.append(metrics)
            series_list.append(coeff)
        
        return summary_list, series_list
    
    def plot_summary(self, summary_list, series_list, title=""):
        """
        Plot summary statistics.
        
        Args:
            summary_list (list): List of summary series
            series_list (list): List of series labels
            title (str): Plot title
        """
        plot_df = []
        
        for i, summary in enumerate(summary_list):
            temp_df = pd.DataFrame({
                "x": summary.index,
                "y": summary.values,
                "series": f"{series_list[i]} MAE coefficient"
            })
            plot_df.append(temp_df)
        
        plot_df = pd.concat(plot_df, ignore_index=True)
        
        g = sns.catplot(
            data=plot_df,
            x="x",
            y="y",
            hue="series",
            kind="bar",
            height=6,
            aspect=1.5
        )
        
        g.set_xticklabels(rotation=45)
        g.fig.suptitle(title)
        g.savefig(f"{self.reports_dir}{title}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary plot saved: {self.reports_dir}{title}.png")


# Example usage:

def run_log_analysis(dataset_name, start_version, end_version, log_dir=None):



    analyzer = LogAnalyzer(
        dataset_name=dataset_name,
        log_dir=log_dir,
        start_version=start_version,
        end_version=end_version

    )
    # Export logs to CSV
    analyzer.export_logs_to_csv()
        
    # load existing report
    if analyzer.load_report() is None:
        analyzer.generate_report()
        
    
    # Create visualizations
    analyzer.plot_error_analysis()
    
    # Create pivot table
    pivot = analyzer.create_pivot_table()
    print(pivot)

if __name__ == "__main__":
    pass
    # Initialize analyzer
    # Dataset configuration
    #%%
    DATASET_CONFIG = {
        "dichasus": {
            "log_dir": "/home/sharghif/localization/lightning_logs_results/",
            "versions": list(range(100, 145))
        },
        "5G": {
            "log_dir": "/home/sharghif/localization/lightning_logs_results/",
            "versions": list(range(145, 305))
        },
        "5G_2": {
            "log_dir": "/home/sharghif/localization/lightning_logs/",
            "versions": list(range(1, 161))
        },
        "5G_triplet": {
            "log_dir": "/home/sharghif/localization/lightning_logs/",
            "versions": list(range(161, 171))
        },
        "5G_PCGRAD_3": {
            "log_dir": "/home/sharghif/localization/lightning_logs/",
            "versions": list(range(174, 189))
        },
        "5G_PCGRAD": {
            "log_dir": "/home/sharghif/localization/lightning_logs_results/",
            "versions": list(range(467, 482))
        },
        "dichasus_PCGRAD": {
            "log_dir": "/home/sharghif/localization/lightning_logs_results/",
            "versions": list(range(452, 467))
        },
        "dichasus_2": {
            "log_dir": "/home/sharghif/localization/lightning_logs",
            "versions": list(range(209, 224))
        }
    }
    name = "dichasus_2"

    c = DATASET_CONFIG[name]
    start_version = c["versions"][0]
    end_version = c["versions"][-1] + 1
    
    analyzer = LogAnalyzer(dataset_name=name, log_dir=Path(c["log_dir"]), start_version=start_version, end_version=end_version)
    
    # Export logs to CSV
    analyzer.export_logs_to_csv()
    
    # Generate report
    analyzer.generate_report()
    
    # Or load existing report
    # analyzer.load_report()
    
    # Create visualizations
    analyzer.plot_error_analysis(post_fix="100")
    
    # Create pivot table
    pivot = analyzer.create_pivot_table()
    print(pivot)
    

        
# Run through `python main.py analyze`; see `python main.py analyze --help`.
