# coding: utf-8
# MAT Framework libraries. The libraries installed from the Network Workflow configuration must be imported inside each function.
from datetime import datetime
from airflow import DAG
from mat import *
from mat_runtime import *
from airflow.operators import MATInstanceInitOperator, MATInstanceExitOperator, MATPythonOperator, DummyOperator, MATBranchOperator, MATAnsibleBranchOperator, MATAnsiblePlaybookOperator
from mat_error_management import MATWorkflowErrorManagement
from mat_success_management import MATWorkflowSuccessManagement
import json

# Dag definition
default_args = {
    'owner': 'Iquall',
    'depends_on_past': False,
    'provide_context': True
}

dag = DAG(dag_id='9yd-1n5-vuz', description='', start_date=datetime(2026,2,5), schedule_interval=None, catchup=False, on_failure_callback=MATWorkflowErrorManagement, on_success_callback=MATWorkflowSuccessManagement, default_args=default_args)



def _fmt_num(v, decimals=2):
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)

def _fmt_int(v):
    if v is None:
        return "N/A"
    try:
        return f"{int(round(float(v)))}"
    except Exception:
        return str(v)

def task_generate_report(**kwargs):
    '''
    Generate a Job Report by calling /usr/bin/pdflatex directly.
    '''
    try:
        import usecase
    except ImportError:
        pass
    
    import mock
    import resources
    import aggregations
    
    import os
    import shutil
    import glob
    import subprocess
    import json
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
    from datetime import datetime, timedelta, timezone
    import pandas as pd
    
    from mat import MATClient
    
    from utils import get_dc_metrics, get_cluster_metrics, get_vnf_metrics
    from utils import get_server_info, top5_cpu_p95, get_vnf_types, get_vms_info
    from utils import get_vnfc_metrics, get_redundancy_info
    from utils import get_nas_metrics, build_top10_idle_tables, get_vm_disk_metrics
    from utils import get_server_disk_metrics, get_cluster_spare_active_metrics
    from utils import _run_pdflatex_and_log
    
    from utils import build_top10_idle_tables_vnfc, build_top10_cpu_p95, build_geo_redundancy_rows
    from utils import get_dc_monthly
    
    from graph_utils import normalize_vm_disk_metrics, normalize_server_disk_metrics, compute_disk_metrics
    from graph_utils import normalize_cluster_metrics, aggregate_cpu_mem, build_combined, build_res_data
    from graph_utils import plot_fig1_donuts, plot_fig2_breakdown, plot_fig3_normalized, plot_top10_idle_cpu
    from graph_utils import plot_top10_idle_cpu_updated, plot_top10_idle_mem_updated, plot_top10_idle_disk_updated
    from graph_utils import plot_top10_idle_mem, plot_top10_idle_disk, plot_vm_matrix_heatmap, build_capacity_reserve_df
    from graph_utils import plot_capacity_reserve, plot_effective_util_vs_reserve_by_cluster, plot_cluster_cpu_usage
    from graph_utils import get_server_metrics, plot_fig5_servers_per_cluster
    
    from graph_utils import filter_res_data_by_dc, plot_fig2_breakdown_by_dc, aggregate_res_data_by_dc
    from graph_utils import plot_fig5_servers_per_dc
    from graph_utils import build_capacity_reserve_df_multi_dc, plot_vm_matrix_heatmap_by_dc
    from graph_utils import plot_fig3_normalized_by_dc



    job = kwargs.get("job")
    params = getForm(kwargs).get('data',{})
    mat_client = MATClient()
    work_dir = getEnvironment(kwargs, "WORKFLOW_RUN_FOLDER")
    
    # Get date for metrics
    """now = datetime.now(timezone.utc)
    if now.month == 1:
        year = now.year - 1
        month = 12
    else:
        year = now.year
        month = now.month - 1
    dt = datetime(
        year=year,
        month=month,
        day=1,
        hour=22,
        minute=0,
        second=0,
        tzinfo=timezone.utc
    )
    
    metrics_date = dt.isoformat()"""
    metrics_date = "2026-01-23T10:00:00+00:00"
    
    #region = str(params.get('region'))
    region = "9"
    since = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S")
    
    try:
        sites = resources.get_region_sites(job, region)
        dc_metrics = resources.get_datacenter_metrics(job, sites, since)
        region_metrics = resources.get_region_metrics(job, region, since)
        cluster_metrics = resources.get_cluster_metrics(job, sites, since)
        server_info = resources.get_server_info(job, sites)
        vnf_types = resources.get_vnf_types(job, sites)
        vnf_metrics = resources.get_vnf_metrics(job, sites, since)
        vnfc_metrics = resources.get_vnfc_metrics(job, sites, since)
        
        vm_info = resources.get_vms_info(job, sites)
        redundancy_info = resources.get_redundancy_info(job, region)
        
        
        redundancy_info_processed = build_geo_redundancy_rows(job, redundancy_info, dc_names=sites)
        
        nas_metrics, total_nas_metrics = resources.get_nas_metrics(job, sites)
        vm_disk_metrics = resources.get_vm_disk_metrics(job, sites, since)
        server_disk_metrics = resources.get_server_disk_metrics(job, sites, since)
        cluster_spare_active_metrics = resources.get_cluster_spare_active_metrics(job, sites, since)
            
        
        overview_list, overview_total = aggregations.create_overview_info(cluster_metrics, server_info, vnf_types, vm_info)
    

        
        # # =========================
        # # Generación de Gráficos
        # # =========================
        
        vm_df = normalize_vm_disk_metrics(job, vm_disk_metrics)
        server_df = normalize_server_disk_metrics(job, server_disk_metrics)
        disk_metrics, _ = compute_disk_metrics(job, vm_df, server_df)
        
        cluster_df = normalize_cluster_metrics(job, cluster_spare_active_metrics)
        combined_cpu_mem = aggregate_cpu_mem(job, cluster_df)
        combined = build_combined(job, combined_cpu_mem, disk_metrics)
        res_data = build_res_data(job, combined)
        
    
        
        
        top10_vnfc_tables = build_top10_idle_tables_vnfc(job, vnfc_metrics, sites)
        
        top10_vnf_cpu_p95_region = build_top10_cpu_p95(vnf_metrics)
        
        
    
        figs_dir = os.path.join(work_dir, "figs")
        os.makedirs(figs_dir, exist_ok=True)
    
    
        fig1_all_path = os.path.join(figs_dir, "all_dcs_fig1.png")
        plot_fig1_donuts(job, res_data, fig1_all_path)
        
        #plot breakdown by DC
        fig2_by_dc_path = os.path.join(figs_dir, "all_dcs_fig2_by_dc.png")
        plot_fig2_breakdown_by_dc(job, res_data, fig2_by_dc_path)
        
        # Plot servers per DC
        fig5_by_dc_path = os.path.join(figs_dir, "all_dcs_fig5_servers_by_dc.png")
        plot_fig5_servers_per_dc(job, res_data, server_info, fig5_by_dc_path)
    
        fig3_by_dc_path = os.path.join(figs_dir, "all_dcs_fig3_normalized_by_dc.png")
        dc_metrics_monthly = get_dc_monthly(region,mat_client)
        plot_fig3_normalized_by_dc(job, res_data, fig3_by_dc_path,dc_monthly_metrics=dc_metrics_monthly)
        
        
        top_cpu_path = os.path.join(figs_dir, f"{region}_top_cpu.png")
        plot_top10_idle_cpu_updated(job, top10_vnfc_tables["top10_idle_cpu"], top_cpu_path)
        
        top_mem_path = os.path.join(figs_dir, f"{region}_top_mem.png")
        plot_top10_idle_mem_updated(job, top10_vnfc_tables["top10_idle_mem"], top_mem_path)
        
        top_disk_path = os.path.join(figs_dir, f"{region}_top_disk.png")
        plot_top10_idle_disk_updated(job, top10_vnfc_tables["top10_idle_disk"], top_disk_path)
        
        # Plot DC x VNF heatmap (VM counts)
        fig_dc_vnf_matrix_path = os.path.join(figs_dir, "all_dcs_dc_vnf_matrix.png")
        plot_vm_matrix_heatmap_by_dc(job, vm_info, dc_metrics, fig_dc_vnf_matrix_path)
    
         # Plot capacity reserve (CPU + STORAGE) for all DCs
        cap_df = build_capacity_reserve_df_multi_dc(job, dc_metrics)
        cap_cpu_storage_df = cap_df[(cap_df["type"] == "CPU") | (cap_df["type"] == "STORAGE")].copy()
        fig_cpu_storage_path = os.path.join(figs_dir, "all_dcs_cpu_storage_util_vs_reserve.png")
        plot_capacity_reserve(
            job,
            cap_cpu_storage_df,
            x_col="efect_util_pct",
            y_col="efect_resv_pct",
            label_col="DC",
            title="Utilizacion Efectiva vs Reserva Efectiva (CPU/Storage)",
            color_by="type",
            show_labels=True,
            x_label="Utilizacion Efectiva (%)",
            y_label="Reserva Efectiva (%)",
            cmap_name="Dark2",
            save_path=fig_cpu_storage_path,
            new_threshold=None
        )
        
        # Plot capacity reserve (RAM) for all DCs
        cap_mem_df = cap_df[cap_df["type"] == "RAM"].copy()
        fig_mem_path = os.path.join(figs_dir, "all_dcs_mem_util_vs_reserve.png")
        plot_capacity_reserve(
            job,
            cap_mem_df,
            x_col="efect_util_pct",
            y_col="efect_resv_pct",
            label_col="DC",
            title="Utilizacion Efectiva vs Reserva Efectiva (RAM)",
            color_by="type",
            show_labels=True,
            x_label="Utilizacion Efectiva RAM (%)",
            y_label="Reserva Efectiva RAM (%)",
            cmap_name="Dark2",
            save_path=fig_mem_path,
            new_threshold=True
        )
    
        
        # paths = plot_effective_util_vs_reserve_by_cluster(job, cluster_metrics, site, figs_dir)
        # cpu_scatter_path = paths["cpu_path"]
        # mem_scatter_path = paths["mem_path"]
        
        # server_metrics_df = get_server_metrics(job, mat_client, [site], days=14)
    
        # fig_cpu_cluster_path = os.path.join(figs_dir, f"{site}_cluster_cpu_usage.png")
        
        # plot_cluster_cpu_usage(
        #     job,
        #     server_metrics_df,
        #     save_path=fig_cpu_cluster_path
        # )
    
        # # ====================================
        # # Construcción de context para el .tex
        # # ====================================
        
        pdflatex_bin = '/usr/bin/pdflatex'
    
        
        #template_path = f'/usr/local/iquall/mat/shared/sandbox/apps/8o3-29i-fr6/executive_summary.tex.j2'
        #work_dir = os.path.dirname(template_path)
        
        work_dir = os.path.dirname(os.path.realpath(__file__)) 
        os.chdir(work_dir)
        template_path = f'{work_dir}/executive_summary.tex.j2'
    
        template_file = os.path.basename(template_path)
        base_name = template_file
        if base_name.endswith(".tex.j2"):
            base_name = base_name[:-len(".tex.j2")]
        else:
            base_name = os.path.splitext(base_name)[0]  # fallback
    
        rendered_tex_path = os.path.join(work_dir, f"{base_name}.tex")
        expected_pdf_path = os.path.join(work_dir, f"{base_name}.pdf")
        expected_log_path = os.path.join(work_dir, f"{base_name}.log")
    
        job.log.info(f"Starting PDF generation using binary: {pdflatex_bin}")
        job.log.info(f"Processing template: {template_path}")
    
    
        
        context = {
            "region": region,
            "dc_list_str": ", ".join(sites).replace("_"," "),
            "overview_list": overview_list,
            "total_overview": overview_total,
            "region_metrics": dc_metrics,
            "total_region_metric": region_metrics[0],
            "nas_storages": nas_metrics,
            "total_nas_storage": total_nas_metrics,
            "vm_utilization": dc_metrics,
            "avg_vm_utilization": region_metrics[0],
            "top10_vnf_cpu_p95_region": top10_vnf_cpu_p95_region.to_dict(orient="records"),
            "top10_idle_cpu": top10_vnfc_tables["top10_idle_cpu"],
            "top10_idle_mem": top10_vnfc_tables["top10_idle_mem"],
            "top10_idle_disk": top10_vnfc_tables["top10_idle_disk"],
            "redundancy_table": redundancy_info_processed,
            "fig1_path": fig1_all_path,
            "fig2_path": fig2_by_dc_path,
            "fig5_path": fig5_by_dc_path,
            "fig3_path": fig3_by_dc_path,
            "fig_vnf_dc_distribution": fig_dc_vnf_matrix_path,
            "top_cpu_path": top_cpu_path,
            "top_mem_path": top_mem_path,
            "top_disk_path": top_disk_path,
            "fig4_cpu_path": fig_cpu_storage_path,
            "fig4_mem_path": fig_mem_path
        }
    
        job.log.info("Rendering LaTeX template with DB results...")
        job.log.debug(json.dumps(context, indent=2))
    
        env = Environment(
            loader=FileSystemLoader(work_dir),
            undefined=StrictUndefined,
            autoescape=False,
            block_start_string=r'\BLOCK{',
            block_end_string='}',
            variable_start_string=r'\VAR{',
            variable_end_string='}',
            comment_start_string=r'\#{',
            comment_end_string='}',
        )
        
        env.filters["fmt_num"] = _fmt_num
        env.filters["fmt_int"] = _fmt_int
    
        template = env.get_template(template_file)
        rendered_tex = template.render(**context)
        
        job.log.debug("PDF_RENDER")
        job.log.debug(rendered_tex)
        
        with open(rendered_tex_path, "w", encoding="utf-8") as f:
            f.write(rendered_tex)
            
    
        cmd = [pdflatex_bin, "-interaction=nonstopmode", "-output-directory", work_dir, rendered_tex_path] 
       
        job.log.info("Starting PDF generation (Run 1/2)...") 
       
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=work_dir, check=False) 
       
        job.log.info("Starting PDF generation (Run 2/2)...") 
       
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=work_dir, check=False) 
       
        console_output = process.stdout.decode("utf-8", errors="replace") 
       
        if os.path.exists(expected_log_path): 
            with open(expected_log_path, "r", encoding="utf-8", errors="replace") as f: 
                log_content = f.read() 
        else: 
            log_content = console_output
    
        if not os.path.exists(expected_pdf_path):
            job.log.error(f"pdflatex failed with return code {process.returncode}")
            job.log.error(f"Console Output: {console_output}")
            raise Exception(f"Couldn't find expected file: {expected_pdf_path=}")
    
        with open(expected_pdf_path, "rb") as f:
            pdf_encoded = f.read()
    
        mat_add_row_report(
            report="General Report",
            section=f"Region {region}",
            fields={
                "pdf": {
                    "type": "bin_file",
                    "filename": f"Reporte_R{region}.pdf",
                    "value": pdf_encoded,
                },
            },
        )
        
    finally:
        try:
            job = kwargs.get("job")
            work_dir = getEnvironment(kwargs, "WORKFLOW_RUN_FOLDER")

            # 1. Borrar directorio de figuras
            figs_dir = os.path.join(work_dir, "figs")
            if os.path.exists(figs_dir):
                shutil.rmtree(figs_dir, ignore_errors=True)
                job.log.info(f"Deleted figs directory: {figs_dir}")

            # 2. Borrar archivos generados por LaTeX
            patterns = [
                "*.aux", "*.log", "*.out", "*.toc",
                "*.tex", "*.pdf"
            ]

            for pattern in patterns:
                for f in glob.glob(os.path.join(work_dir, pattern)):
                    try:
                        os.remove(f)
                        job.log.debug(f"Deleted file: {f}")
                    except Exception as e:
                        job.log.warning(f"Could not delete {f}: {e}")

        except Exception as e:
            # Nunca debe romper el job por limpieza
            job.log.warning(f"Cleanup failed: {e}")
        
    
# Tasks: Define each task of the dag
init = MATInstanceInitOperator(task_id='MAT_Initialize', dag=dag)
generate_report = MATPythonOperator(task_id="generate_report", python_callable=task_generate_report, dag=dag)
end = MATInstanceExitOperator(task_id='MAT_Finalize', dag=dag)

# Task Workflow: Define task dependencies and workflow execution order
init >> generate_report  >> end