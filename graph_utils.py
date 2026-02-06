def normalize_vm_disk_metrics(job, vm_disk_metrics):
    """
    Build vm_df compatible with the notebook logic:
      - Server Name
      - VM ID
      - Capacity (bytes-ish or MB/GB; we keep raw units as-is)
      - Allocation
    Notes:
      - Your vm_disk_metrics has per-disk entries; we keep per-entry and later aggregate by Server Name.
    """
    import pandas as pd

    rows = []
    for r in vm_disk_metrics or []:
        vm = (r.get("vm") or {})
        hyp = (vm.get("hypervisor_vm") or {})
        server_name = hyp.get("name")
        
        total_mb = r.get("total")
        used_mb = r.get("used")

        cap_bytes = float(total_mb) * (1024**2) if total_mb is not None else 0.0
        alloc_bytes = float(used_mb) * (1024**2) if used_mb is not None else 0.0

        rows.append(
            {
                "Server Name": server_name,
                "VM ID": vm.get("name"),
                "disk": r.get("disk"),
                "Capacity": cap_bytes,
                "Allocation": alloc_bytes,
                "time": r.get("time"),
            }
        )

    df = pd.DataFrame(rows)
    job.log.debug(f"[normalize_vm_disk_metrics] rows={len(df)} cols={list(df.columns)}")

    # Basic cleanup
    if not df.empty:
        df["Server Name"] = df["Server Name"].astype(str).str.strip()
        df["VM ID"] = df["VM ID"].astype(str).str.strip()

    return df

def normalize_server_disk_metrics(job, server_disk_metrics):
    """
    Build server_df compatible with the notebook logic:
      - Server Name
      - Availability Zone (cluster)
      - Total Physical Capacity  (WE WILL USE server.storage_total_gb as requested)
      - System Reserved          (storage_system)
    Also keeps a few extra fields used elsewhere in the original notebook.
    """
    import pandas as pd

    rows = []
    for r in server_disk_metrics or []:
        s = (r.get("server") or {})
        cluster = (s.get("cluster_server") or {}).get("name")
        
        storage_total_gb = s.get("storage_total_gb")
        storage_system_gb = r.get("storage_system")

        # Convert GB -> bytes
        total_phys_bytes = float(storage_total_gb) * (1024**3) if storage_total_gb is not None else 0.0
        system_res_bytes = float(storage_system_gb) * (1024**3) if storage_system_gb is not None else 0.0


        rows.append(
            {
                "Server Name": s.get("hostname"),
                "Availability Zone": cluster,
               "Total Physical Capacity": total_phys_bytes,
                "System Reserved": system_res_bytes,
                # extra (not required for disk_metrics but helpful for validation):
                "VM Storage Used": r.get("storage_used"),
                "VM Storage Reserved": r.get("storage_reserved"),
                "VM Storage Free": r.get("storage_free"),
                "storage_total_raw": r.get("storage_total"),
                "time": r.get("time"),
            }
        )

    df = pd.DataFrame(rows)
    job.log.debug(f"[normalize_server_disk_metrics] rows={len(df)} cols={list(df.columns)}")

    if not df.empty:
        df["Server Name"] = df["Server Name"].astype(str).str.strip()
        df["Availability Zone"] = df["Availability Zone"].astype(str).str.strip()

    return df

def compute_disk_metrics(job, vm_df, server_df):
    """
    Replicates the notebook disk aggregation logic as closely as possible.

    Outputs:
      - disk_metrics: indexed by Availability Zone (cluster), with *_tb columns:
          disk_reserved_tb, disk_used_tb, disk_total_tb, disk_sys_reserved_tb,
          plus their original byte-like columns if needed.
      - merged_df: server-level join used by some checks/plots in the notebook
          (includes VM_Sum_Allocation, VM_Sum_Capacity, VM_Count)
    """
    import numpy as np
    import pandas as pd

    disk_metrics = pd.DataFrame()
    merged_df = pd.DataFrame()

    if server_df is None or server_df.empty:
        job.log.warning("[compute_disk_metrics] server_df empty; disk_metrics will be empty.")
        return disk_metrics, merged_df

    # Ensure required columns exist / numeric
    server_df = server_df.copy()

    server_df["System Reserved"] = pd.to_numeric(server_df.get("System Reserved"), errors="coerce").fillna(0)
    server_df["Total Physical Capacity"] = pd.to_numeric(
        server_df.get("Total Physical Capacity"), errors="coerce"
    ).fillna(0)

    if "Availability Zone" not in server_df.columns:
        # In your latest payload it exists; keep same fallback as notebook.
        server_df["Availability Zone"] = "Unknown"

    # VM part
    if vm_df is None or vm_df.empty:
        job.log.warning("[compute_disk_metrics] vm_df empty; will aggregate only server totals/system.")
        # still can produce disk_total_tb and disk_sys_reserved_tb by AZ
        server_agg = (
            server_df.groupby("Availability Zone")
            .agg({"Total Physical Capacity": "sum", "System Reserved": "sum"})
            .rename(
                columns={
                    "Total Physical Capacity": "disk_total_bytes",
                    "System Reserved": "disk_sys_reserved_bytes",
                }
            )
        )
        disk_stats = server_agg.fillna(0)
        for c in list(disk_stats.columns):
            disk_stats[c.replace("bytes", "tb")] = disk_stats[c] / (1024**4)
        return disk_stats, merged_df

    vm_df = vm_df.copy()
    vm_df["Capacity"] = pd.to_numeric(vm_df.get("Capacity"), errors="coerce").fillna(0)
    vm_df["Allocation"] = pd.to_numeric(vm_df.get("Allocation"), errors="coerce").fillna(0)

    # Notebook adds Utilization_Pct; not strictly needed for our 3 main figures,
    # but keep it to remain compatible with their downstream analysis.
    vm_df["Utilization_Pct"] = np.where(vm_df["Capacity"] > 0, (vm_df["Allocation"] / vm_df["Capacity"]) * 100, 0)

    # Enrich VM with server totals + AZ
    req_cols = ["Server Name", "Total Physical Capacity", "Availability Zone"]
    for c in req_cols:
        if c not in server_df.columns:
            server_df[c] = 0

    vm_enriched = pd.merge(vm_df, server_df[req_cols], on="Server Name", how="left")

    # NAS filter exactly like notebook:
    nas_mask = vm_enriched["Capacity"] > vm_enriched["Total Physical Capacity"]
    local_disks = vm_enriched[~nas_mask].copy()

    # server-level merged_df (used by notebook checks / some plots)
    vm_agg = (
        vm_df.groupby("Server Name")
        .agg({"Allocation": "sum", "Capacity": "sum", "VM ID": "nunique"})
        .rename(
            columns={
                "Allocation": "VM_Sum_Allocation",
                "Capacity": "VM_Sum_Capacity",
                "VM ID": "VM_Count",
            }
        )
        .reset_index()
    )
    merged_df = pd.merge(server_df, vm_agg, on="Server Name", how="left")
    merged_df["VM_Sum_Allocation"] = merged_df["VM_Sum_Allocation"].fillna(0)
    merged_df["VM_Sum_Capacity"] = merged_df["VM_Sum_Capacity"].fillna(0)
    merged_df["VM_Count"] = merged_df["VM_Count"].fillna(0)

    # AZ-level disk stats (this is what the notebook injects into `combined`)
    disk_agg = local_disks.groupby("Availability Zone").agg({"Capacity": "sum", "Allocation": "sum"})
    disk_agg.columns = ["disk_reserved_bytes", "disk_used_bytes"]

    server_agg = (
        server_df.groupby("Availability Zone")
        .agg({"Total Physical Capacity": "sum", "System Reserved": "sum"})
        .rename(
            columns={
                "Total Physical Capacity": "disk_total_bytes",
                "System Reserved": "disk_sys_reserved_bytes",
            }
        )
    )

    disk_stats = pd.concat([disk_agg, server_agg], axis=1).fillna(0)

    # Convert to TB exactly like notebook (divide by 1024**4)
    for c in list(disk_stats.columns):
        disk_stats[c.replace("bytes", "tb")] = disk_stats[c] / (1024**4)

    disk_metrics = disk_stats

    job.log.debug(
        "[compute_disk_metrics] "
        f"local_disks_rows={len(local_disks)} "
        f"nas_rows={int(nas_mask.sum())} "
        f"az_count={disk_metrics.shape[0]}"
    )

    return disk_metrics, merged_df

def normalize_cluster_metrics(job, cluster_spare_active_metrics):
    """
    Build DataFrame equivalent to cluster_new.csv
    (one row per cluster per role: active/spare).
    """
    import pandas as pd

    df = pd.DataFrame(cluster_spare_active_metrics or [])
    job.log.debug(f"[normalize_cluster_metrics] rows={len(df)} cols={list(df.columns)}")

    if df.empty:
        return df

    # Extract dc (datacenter name) from nested dict if present
    if "datacenter" in df.columns:
        df["dc"] = df["datacenter"].apply(lambda x: x.get("name", "") if isinstance(x, dict) else "")

    # Normalize has_vms to bool (not int)
    df["has_vms"] = df["has_vms"].apply(lambda x: True if int(x) == 1 else False)

    # Ensure numeric cols
    numeric_cols = [
        "vcpu_capacity",
        "cpu_cores",
        "cpu_reserved",
        "mem_capacity",
        "mem_reserved",
        "cpu_p95_pct",
        "mem_p95_pct",
        "cpu_fragmented",
        "mem_reseverd_system",
        "disk_capacity",
        "cpu_util_efect",
        "mem_util_efect",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["cpu_fragmented"] = df["cpu_fragmented"].apply(lambda x: 0 if x < 0 else x)
    return df

def aggregate_cpu_mem(job, cluster_df):
    """
    Replicates the active/spare aggregation logic from the notebook.
    Output: combined_cpu_mem (NO disk yet)
    """
    import pandas as pd

    if cluster_df.empty:
        job.log.warning("[aggregate_cpu_mem] cluster_df empty")
        return pd.DataFrame()

    # Preserve cluster -> dc mapping (each cluster belongs to one DC)
    dc_map = None
    if "dc" in cluster_df.columns:
        dc_map = cluster_df.groupby("cluster")["dc"].first().reset_index()

    # Split active / spare
    active = cluster_df[cluster_df["has_vms"] == True].copy()
    spare = cluster_df[cluster_df["has_vms"] == False].copy()

    # Patch: system reserved >= capacity → 6.5%
    mask_active = active["mem_reseverd_system"] >= active["mem_capacity"]
    if mask_active.any():
        active.loc[mask_active, "mem_reseverd_system"] = active.loc[mask_active, "mem_capacity"] * 0.065

    mask_spare = spare["mem_reseverd_system"] >= spare["mem_capacity"]
    if mask_spare.any():
        spare.loc[mask_spare, "mem_reseverd_system"] = spare.loc[mask_spare, "mem_capacity"] * 0.065

    # Columns exactly like notebook
    cols_base = [
        "vcpu_capacity",
        "cpu_cores",
        "cpu_reserved",
        "mem_capacity",
        "mem_reserved",
        "cpu_p95_pct",
        "mem_p95_pct",
        "cpu_fragmented",
        "mem_reseverd_system",
        "disk_capacity",
        "cpu_util_efect",
        "mem_util_efect",
    ]

    cols_active = [c for c in cols_base if c in active.columns]
    cols_spare = [c for c in cols_base if c in spare.columns]

    # Sum active
    active_agg = active.groupby("cluster")[cols_active].sum().reset_index()

    # Max for efect utilization (igual que notebook)
    max_cols = [c for c in ["cpu_util_efect", "mem_util_efect"] if c in active.columns]
    if max_cols:
        max_stats = active.groupby("cluster")[max_cols].max().reset_index()
        for c in max_cols:
            active_agg[c] = max_stats[c]

    active_agg.columns = ["cluster"] + [f"{c}_active" for c in cols_active]

    # Sum spare
    spare_agg = spare.groupby("cluster")[cols_spare].sum().reset_index()
    spare_agg.columns = ["cluster"] + [f"{c}_spare" for c in cols_spare]

    # Merge
    combined = pd.merge(active_agg, spare_agg, on="cluster", how="outer").fillna(0)

    # Restore dc column
    if dc_map is not None:
        combined = pd.merge(combined, dc_map, on="cluster", how="left")

    job.log.debug(
        "[aggregate_cpu_mem] "
        f"clusters={len(combined)} cols={list(combined.columns)}"
    )

    return combined

def build_combined(job, combined_cpu_mem, disk_metrics):
    """
    combined_cpu_mem: DF con columnas *_active/*_spare por cluster
    disk_metrics: DF indexado por Availability Zone (cluster) con disk_*_tb
    """
    import pandas as pd

    df = combined_cpu_mem.copy()

    # Inject disk_metrics columns into combined (match notebook: combined[c] = disk_metrics[c])
    if disk_metrics is not None and not disk_metrics.empty:
        dm = disk_metrics.copy()

        # disk_metrics index is Availability Zone; make it a column named 'cluster' for merge
        dm = dm.reset_index().rename(columns={"Availability Zone": "cluster"})

        # keep only the TB columns used later (same as notebook uses)
        keep = ["cluster"]
        for c in ["disk_total_tb", "disk_sys_reserved_tb", "disk_reserved_tb", "disk_used_tb"]:
            if c in dm.columns:
                keep.append(c)

        dm = dm[keep]

        df = pd.merge(df, dm, on="cluster", how="left")

        # Fill missing disk columns with 0
        for c in ["disk_total_tb", "disk_sys_reserved_tb", "disk_reserved_tb", "disk_used_tb"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Fill remaining NaNs
    df = df.fillna(0)

    # Notebook: remove internal
    if "cluster" in df.columns:
        excludes = ["internal", "unknown"]
        pattern = "|".join(excludes)
        # Use str.contains with case=False to exclude any matching cluster names
        df = df[~df['cluster'].str.contains(pattern, case=False, na=False)].copy()

    job.log.debug(f"[build_combined] rows={len(df)} cols={list(df.columns)}")
    return df

def build_res_data(job, combined):
    """
    Returns:
      res_data dict with keys: 'vCPU', 'Memoria', 'Disco'
      Each entry: {'df': df, 'unit': unit}
    """
    import numpy as np

    res_data = {}
    resources = ["vCPU", "Memoria", "Disco"]

    for res in resources:
        df = combined.copy()

        if res == "vCPU":
            unit = "vCPUs"

            cores_total_active = df["cpu_cores_active"] * 2
            cap_net_active = df["vcpu_capacity_active"]
            sys_active = (cores_total_active - cap_net_active).clip(lower=0)

            cores_total_spare = df["cpu_cores_spare"] * 2
            cap_net_spare = df["vcpu_capacity_spare"]
            sys_spare = (cores_total_spare - cap_net_spare).clip(lower=0)

            df["v_sys"] = sys_active + sys_spare
            df["v_spare"] = cap_net_spare
            df["Total_Cap"] = cores_total_active + cores_total_spare

            res_active = df["cpu_reserved_active"]
            p95_pct = df["cpu_p95_pct_active"]
            p95_abs = res_active * (p95_pct / 100.0)
            consumption_peak = np.maximum(res_active, p95_abs)

            df["v_frag"] = df["cpu_fragmented_active"]
            df["v_res"] = res_active
            #df["v_res"] = consumption_peak
            df["v_free"] = (df["Total_Cap"] - df["v_sys"] - df["v_frag"] - df["v_res"] - df["v_spare"]).clip(lower=0)
            df["real_used_abs"] = 0
            df["p95_abs"] = p95_abs
            
            job.log.debug(f"********************* BUILD_RES_DATA - vCPU *********************")
            job.log.debug(df.to_dict(orient="records"))

        elif res == "Memoria":
            unit = "GB"

            sys_res = df.get("mem_reseverd_system_active", 0) + df.get("mem_reseverd_system_spare", 0)
            df["Total_Cap"] = df["mem_capacity_active"] + df["mem_capacity_spare"]

            res_active = df["mem_reserved_active"]
            p95_pct = df["mem_p95_pct_active"]
            p95_abs = res_active * (p95_pct / 100.0)
            consumption_peak = np.maximum(res_active, p95_abs)

            df["v_sys"] = sys_res
            df["v_frag"] = 0
            df["v_res"] = consumption_peak
            df["v_spare"] = df["mem_capacity_spare"] - df.get("mem_reseverd_system_spare", 0)
            df["v_free"] = (df["Total_Cap"] - consumption_peak - df["v_spare"] - sys_res).clip(lower=0)
            df["p95_abs"] = p95_abs
            df["real_used_abs"] = 0

        elif res == "Disco":
            unit = "TB"

            raw_total = df.get("disk_capacity_active", 0) + df.get("disk_capacity_spare", 0)
            ratio = np.where(raw_total > 0, df.get("disk_capacity_spare", 0) / raw_total, 0)

            phys_total = df["disk_total_tb"]
            phys_sys = df["disk_sys_reserved_tb"]
            phys_net = (phys_total - phys_sys).clip(lower=0)

            df["v_sys"] = phys_sys
            df["v_res"] = df["disk_reserved_tb"]
            df["v_spare"] = phys_net * ratio
            active_net = phys_net * (1 - ratio)
            df["v_free"] = (active_net - df["v_res"]).clip(lower=0)
            df["v_frag"] = 0
            df["Total_Cap"] = phys_total
            df["p95_abs"] = df["disk_used_tb"]
            df["real_used_abs"] = df["disk_used_tb"]

        res_data[res] = {"df": df, "unit": unit}   
        
    job.log.debug("[build_res_data] OK: built vCPU/Memoria/Disco dataframes")
    return res_data

def filter_res_data_by_dc(res_data, dc_name):
    """Filter res_data to include only rows for a specific datacenter."""
    filtered = {}
    for res in res_data:
        df_filtered = res_data[res]["df"][res_data[res]["df"]["dc"] == dc_name].copy()
        filtered[res] = {"df": df_filtered}
        if "unit" in res_data[res]:
            filtered[res]["unit"] = res_data[res]["unit"]
    return filtered


def _get_cluster_order(res_data):
    master = res_data["vCPU"]["df"].sort_values(by="Total_Cap")
    return master["cluster"].tolist()

def plot_fig1_donuts(job, res_data, out_path):
    import matplotlib.pyplot as plt

    resources = ["vCPU", "Memoria", "Disco"]

    # Colores (igual al notebook)
    c_sys = "#7f7f7f"
    c_res = "#e67e22"
    c_spare = "#87CEEB"
    c_free = "#2ecc71"
    c_frag = "#8E44AD"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, res in enumerate(resources):
        df = res_data[res]["df"]
        ax = axes[i]

        vals = [df["v_sys"].sum(), df["v_res"].sum(), df["v_frag"].sum(), df["v_spare"].sum(), df["v_free"].sum()]
        labs = ["Sistema", "Reservado", "Fragmentación", "Spare", "Libre"]
        cols = [c_sys, c_res, c_frag, c_spare, c_free]

        v_final, l_final, c_final = [], [], []
        total = float(sum(vals))

        for v, l, c in zip(vals, labs, cols):
            if v and v > 0:
                v_final.append(float(v))
                pct = (float(v) / total * 100.0) if total > 0 else 0.0
                l_final.append(f"{l}: {pct:.1f}%")
                c_final.append(c)

        ax.pie(
            v_final,
            labels=l_final,
            colors=c_final,
            wedgeprops=dict(width=0.3),
            textprops={"fontsize": 13},
        )
        ax.set_title(f"Resumen Global - {res}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    job.log.debug(f"[plot_fig1_donuts] saved: {out_path}")

def plot_fig2_breakdown(job, res_data, out_path):
    import numpy as np
    import matplotlib.pyplot as plt

    resources = ["vCPU", "Memoria", "Disco"]
    order = _get_cluster_order(res_data)

    # Colores (igual notebook)
    c_sys = "#7f7f7f"
    c_res = "#e67e22"
    c_spare = "#87CEEB"
    c_free = "#2ecc71"
    c_frag = "#8E44AD"

    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    handles, labels = None, None

    # Datacenter totals (para % en el label del cluster)
    datacenter_totals = {res: float(res_data[res]["df"]["Total_Cap"].sum()) for res in resources}

    for i, res in enumerate(resources):
        df = res_data[res]["df"].set_index("cluster").reindex(order).reset_index()
        ax = axes[i]

        idx = np.arange(len(df))
        norm = df["Total_Cap"].replace(0, 1)

        p_sys = (df["v_sys"] / norm) * 100.0
        p_spare = (df["v_spare"] / norm) * 100.0
        p_frag = (df["v_frag"] / norm) * 100.0
        p_res = (df["v_res"] / norm) * 100.0
        p_free = (df["v_free"] / norm) * 100.0

        left = np.zeros(len(df))
        ax.barh(idx, p_sys, 0.6, left=left, color=c_sys, label="Sistema")
        left += p_sys
        ax.barh(idx, p_spare, 0.6, left=left, color=c_spare, label="Spare")
        left += p_spare
        if res == "vCPU":
            ax.barh(idx, p_frag, 0.6, left=left, color=c_frag, label="Fragmentación")
            left += p_frag
        ax.barh(idx, p_res, 0.6, left=left, color=c_res, label="Reservado")
        left += p_res
        ax.barh(idx, p_free, 0.6, left=left, color=c_free, label="Libre")
        left += p_free

        ax.set_yticks(idx)

        if i == 0:
            dc_total = datacenter_totals[res]
            cluster_labels = []
            for _, row in df.iterrows():
                cluster_name = row["cluster"]
                cluster_pct = (float(row["Total_Cap"]) / dc_total * 100.0) if dc_total > 0 else 0.0
                cluster_labels.append(f"{cluster_name} ({cluster_pct:.1f}%)")
            ax.set_yticklabels(cluster_labels, fontsize=14)
            ax.text(-0.3, 1.005, "Cluster (% de recursos del DC)", 
                transform=ax.transAxes, 
                fontsize=13, fontweight='bold', color='black',
                ha='center', va='bottom',
                bbox=dict(fc="white", ec="none", alpha=0.0))
        else:
            ax.set_yticklabels([])

        ax.set_xlabel("Porcentaje de Capacidad Total (%)", fontsize=13)
        ax.set_title(f"Desglose de Capacidad - {res}", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05), fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    job.log.debug(f"[plot_fig2_breakdown] saved: {out_path}")


def aggregate_res_data_by_dc(res_data):
    """
    Aggregate res_data by datacenter (dc).
    Returns a new res_data dict where each row is a DC instead of a cluster.
    """
    aggregated = {}
    for res in res_data:
        df = res_data[res]["df"].copy()
        if "dc" not in df.columns:
            # No dc column, return as-is
            aggregated[res] = {"df": df}
            if "unit" in res_data[res]:
                aggregated[res]["unit"] = res_data[res]["unit"]
            continue
        
        # Columns to sum by DC
        sum_cols = ["v_sys", "v_res", "v_frag", "v_spare", "v_free", "Total_Cap"]
        sum_cols = [c for c in sum_cols if c in df.columns]
        
        df_agg = df.groupby("dc")[sum_cols].sum().reset_index()
        aggregated[res] = {"df": df_agg}
        if "unit" in res_data[res]:
            aggregated[res]["unit"] = res_data[res]["unit"]
    
    return aggregated


def _get_dc_order(res_data):
    """Get DC order sorted by Total_Cap (ascending)."""
    master = res_data["vCPU"]["df"].sort_values(by="Total_Cap")
    return master["dc"].tolist()


def plot_fig2_breakdown_by_dc(job, res_data, out_path):
    """
    Plot breakdown chart with one row per datacenter (DC).
    Aggregates cluster data by DC before plotting.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Aggregate by DC
    res_data_dc = aggregate_res_data_by_dc(res_data)
    
    resources = ["vCPU", "Memoria", "Disco"]
    order = _get_dc_order(res_data_dc)

    # Colores (igual notebook)
    c_sys = "#7f7f7f"
    c_res = "#e67e22"
    c_spare = "#87CEEB"
    c_free = "#2ecc71"
    c_frag = "#8E44AD"

    fig, axes = plt.subplots(1, 3, figsize=(24, max(6, len(order) * 0.8)))
    handles, labels = None, None

    # Region totals (para % en el label del DC)
    region_totals = {res: float(res_data_dc[res]["df"]["Total_Cap"].sum()) for res in resources}

    for i, res in enumerate(resources):
        df = res_data_dc[res]["df"].set_index("dc").reindex(order).reset_index()
        ax = axes[i]

        idx = np.arange(len(df))
        #norm = df["Total_Cap"].replace(0, 1)
        
        norm = df["v_sys"] + df["v_spare"] + df["v_frag"] + df["v_res"] +df["v_free"] # Temporal #TODO Revisar
        job.log.debug(f"********************* plot_fig2_breakdown_by_dc - {res} *********************")
        job.log.debug(df.to_dict(orient="records"))
        p_sys = (df["v_sys"] / norm) * 100.0
        p_spare = (df["v_spare"] / norm) * 100.0
        p_frag = (df["v_frag"] / norm) * 100.0
        p_res = (df["v_res"] / norm) * 100.0
        p_free = (df["v_free"] / norm) * 100.0

        left = np.zeros(len(df))
        ax.barh(idx, p_sys, 0.6, left=left, color=c_sys, label="Sistema")
        left += p_sys
        ax.barh(idx, p_spare, 0.6, left=left, color=c_spare, label="Spare")
        left += p_spare
        if res == "vCPU":
            ax.barh(idx, p_frag, 0.6, left=left, color=c_frag, label="Fragmentación")
            left += p_frag
        ax.barh(idx, p_res, 0.6, left=left, color=c_res, label="Reservado")
        left += p_res
        ax.barh(idx, p_free, 0.6, left=left, color=c_free, label="Libre")
        left += p_free

        ax.set_yticks(idx)

        if i == 0:
            region_total = region_totals[res]
            dc_labels = []
            for _, row in df.iterrows():
                dc_name = row["dc"]
                dc_pct = (float(row["Total_Cap"]) / region_total * 100.0) if region_total > 0 else 0.0
                dc_labels.append(f"{dc_name} ({dc_pct:.1f}%)")
            ax.set_yticklabels(dc_labels, fontsize=14)
            ax.text(-0.3, 1.005, "Datacenter (% de recursos de la Región)", 
                transform=ax.transAxes, 
                fontsize=13, fontweight='bold', color='black',
                ha='center', va='bottom',
                bbox=dict(fc="white", ec="none", alpha=0.0))
        else:
            ax.set_yticklabels([])

        ax.set_xlabel("Porcentaje de Capacidad Total (%)", fontsize=13)
        ax.set_title(f"Desglose de Capacidad - {res}", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05), fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    job.log.debug(f"[plot_fig2_breakdown_by_dc] saved: {out_path}")


def plot_fig3_normalized(job, res_data, out_path):
    import numpy as np
    import matplotlib.pyplot as plt

    resources = ["vCPU", "Memoria", "Disco"]
    order = _get_cluster_order(res_data)

    # Colores (igual notebook)
    c_res = "#e67e22"
    c_spare = "#87CEEB"
    c_free = "#2ecc71"
    c_util = "#2980b9"
    c_p95 = "#c0392b"

    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    handles3, labels3 = None, None

    # Datacenter totals (para % en label cluster, igual Fig 2)
    datacenter_totals = {res: float(res_data[res]["df"]["Total_Cap"].sum()) for res in resources}

    for i, res in enumerate(resources):
        df = res_data[res]["df"].set_index("cluster").reindex(order).reset_index()
        ax = axes[i]

        idx = np.arange(len(df))
        w = 0.25

        # V19 Fix: Threshold 0.01 (10GB / 0.01 TB / etc. depende de unidad)
        is_spare_cluster = (df["v_res"] < 0.01) & (df["v_spare"] > 0)

        prov_cap = np.where(
            is_spare_cluster,
            df["v_spare"],
            (df["v_res"] + df["v_free"]),
        ).clip(min=1)

        # Barra 1 - componentes estructura
        p_res = np.where(is_spare_cluster, 0, (df["v_res"] / prov_cap) * 100.0)
        p_free = np.where(is_spare_cluster, 0, (df["v_free"] / prov_cap) * 100.0)
        p_spare = np.where(is_spare_cluster, 100.0, 0.0)

        # Barra 2 - Utilización efectiva / uso local
        if res == "Disco":
            p_util = (df["disk_used_tb"] / prov_cap) * 100.0
        else:
            # V16 Fix: Escalar Utilización Efectiva
            raw_pct = df.get("cpu_util_efect_active", 0) if res == "vCPU" else df.get("mem_util_efect_active", 0)
            reserved_abs = df["v_res"]
            scale_factor = np.where(prov_cap > 0, reserved_abs / prov_cap, 0)
            p_util = raw_pct * scale_factor

        # Barra 3 - P95 (solo CPU y Mem)
        p_p95 = (df.get("p95_abs", 0) / prov_cap) * 100.0

        # Ploteo Barra 1 (idx-w): Res + Spare + Free
        l = np.zeros(len(df))
        ax.barh(idx - w, p_res, w, left=l, color=c_res, label="Reservado")
        l += p_res
        ax.barh(idx - w, p_spare, w, left=l, color=c_spare, label="Spare")
        l += p_spare
        ax.barh(idx - w, p_free, w, left=l, color=c_free, label="Libre")

        # Barra 2 (idx): Utilización / Utilización Efectiva
        if res == "Disco":
            label_util = "Utilización Local"
            color_util = c_p95  # notebook: disk usa color rojo (aunque el comentario diga azul)
        else:
            label_util = "Utilización Efectiva"
            color_util = c_util

        ax.barh(idx, p_util, w, color=color_util, label=label_util)

        # Barra 3 (idx+w): P95 (solo para vCPU y Memoria)
        if res != "Disco":
            ax.barh(idx + w, p_p95, w, color=c_p95, label="Utilización Local")

        ax.set_xlabel("Porcentaje de Capacidad Total Utilizable (%))", fontsize=14)
        ax.set_yticks(idx)

        # Labels cluster solo en el panel izquierdo, con % datacenter (igual Fig 2)
        if i == 0:
            dc_total = datacenter_totals[res]
            cluster_labels = []
            for _, row in df.iterrows():
                cluster_name = row["cluster"]
                cluster_pct = (float(row["Total_Cap"]) / dc_total * 100.0) if dc_total > 0 else 0.0
                cluster_labels.append(f"{cluster_name} ({cluster_pct:.1f}%)")
            ax.set_yticklabels(cluster_labels, fontsize=14)
            ax.text(-0.3, 1.005, "Cluster (% de recursos del DC)", 
                transform=ax.transAxes, 
                fontsize=13, fontweight='bold', color='black',
                ha='center', va='bottom',
                bbox=dict(fc="white", ec="none", alpha=0.0))
        else:
            ax.set_yticklabels([])

        ax.set_title(f"Porcentaje de Utilización - {res}", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        if i == 0:
            handles3, labels3 = ax.get_legend_handles_labels()

    fig.legend(handles3, labels3, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.05), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    job.log.debug(f"[plot_fig3_normalized] saved: {out_path}")

def plot_fig5_servers_per_cluster(job, res_data, server_info, out_path):
    from utils import count_active_spare_servers
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    server_counts_df = count_active_spare_servers(server_info)

    # Prepare Master Data
    master = server_counts_df.copy()
    
    if 'vCPU' in res_data and 'df' in res_data['vCPU']:
        cpu_df = res_data['vCPU']['df']
        if 'Total_Cap' in cpu_df.columns and 'cluster' in cpu_df.columns:
            caps = cpu_df.groupby('cluster')['Total_Cap'].sum().reset_index()
            master = pd.merge(master, caps, on='cluster', how='left')
            master['Total_Cap'] = master['Total_Cap'].fillna(0)
        else:
            master['Total_Cap'] = 0
    else:
        master['Total_Cap'] = 0

    # --- Filtering (Added) ---
    excludes = ["nova", "internal", "unknown"]
    pattern = "|".join(excludes)
    master = master[~master['cluster'].str.contains(pattern, case=False, na=False)]

    # --- Sorting (Added) ---
    # Sort by total servers ascending so largest is at the top of the bar chart
    master['total_servers'] = master['active_server_count'] + master['spare_server_count']
    master = master.sort_values(by='total_servers', ascending=True)

    n_bars = len(master)
    if n_bars == 0:
        job.log.warning("No data to plot for servers per cluster.")
        return

    # Dynamic height
    fig_height = max(4, n_bars * 0.5)
    
    # --- Fix: Assign figure ---
    fig = plt.figure(figsize=(12, fig_height))
    ax = plt.gca()

    y_pos = np.arange(len(master))
    active_vals = master['active_server_count'].to_numpy()
    spare_vals = master['spare_server_count'].to_numpy()

    # Calculate Percentages
    total_dc_cpu = master['Total_Cap'].sum()
    cluster_labels = []
    
    for cl, cap in zip(master['cluster'], master['Total_Cap']):
        pct = (cap / total_dc_cpu) * 100 if total_dc_cpu > 0 else 0
        cluster_labels.append(f"{cl} ({pct:.1f}%)")

    # Plot
    bar_height = 0.6
    p1 = plt.barh(y_pos, active_vals, height=bar_height, color='#00008B', label='Activos')
    p2 = plt.barh(y_pos, spare_vals, left=active_vals, height=bar_height, color='#87CEEB', label='Spare')

    plt.title('Cantidad de Servidores por Cluster', fontsize=13, fontweight='bold')

    ax.text(-0.4, 1.005, "Cluster (% de recursos del DC)", 
            transform=ax.transAxes, fontsize=13, fontweight='bold', 
            va='bottom', ha='left',
            bbox=dict(fc="white", ec="none", alpha=0.0))

    plt.yticks(y_pos, cluster_labels)
    plt.grid(axis='x', alpha=0.3)
    plt.legend(loc='lower right')

    max_val = (active_vals + spare_vals).max()
    if max_val > 0:
        plt.xlim(0, max_val * 1.35)

    for i, (act, spr) in enumerate(zip(active_vals, spare_vals)):
        total = act + spr
        if total > 0:
            label_text = f"Activos: {int(act)} / Spare: {int(spr)}"
            plt.text(total + 0.5, i, label_text, va='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    
    # --- Fix: Close figure ---
    plt.close(fig)

    job.log.debug(f"[plot_fig4_servers_per_cluster] saved: {out_path}")


def count_active_spare_servers_by_dc(server_list):
    """
    Aggregates server list into a DataFrame with columns: 
    ['dc', 'active_server_count', 'spare_server_count']
    """
    import pandas as pd
    if not server_list:
        return pd.DataFrame(columns=['dc', 'active_server_count', 'spare_server_count'])

    rows = []
    for s in server_list:
        # Extract DC from rack_server.dc_rack.name
        dc = "Unknown"
        if s.get("rack_server") and s["rack_server"].get("dc_rack") and s["rack_server"]["dc_rack"].get("name"):
            dc = s["rack_server"]["dc_rack"]["name"]
        
        role = s.get("role", "").lower()
        is_spare = "spare" in role
        
        rows.append({
            "dc": dc,
            "is_spare": is_spare
        })
    
    df = pd.DataFrame(rows)
    
    # Group by dc and count active/spare
    grouped = df.groupby("dc")["is_spare"].agg(
        active_server_count=lambda x: (~x).sum(),
        spare_server_count=lambda x: x.sum()
    ).reset_index()
    
    return grouped


def plot_fig5_servers_per_dc(job, res_data, server_info, out_path):
    """
    Plot server counts per datacenter (DC) instead of per cluster.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    server_counts_df = count_active_spare_servers_by_dc(server_info)

    # Prepare Master Data
    master = server_counts_df.copy()
    
    # Aggregate res_data by DC to get Total_Cap per DC
    if 'vCPU' in res_data and 'df' in res_data['vCPU']:
        cpu_df = res_data['vCPU']['df']
        if 'Total_Cap' in cpu_df.columns and 'dc' in cpu_df.columns:
            caps = cpu_df.groupby('dc')['Total_Cap'].sum().reset_index()
            master = pd.merge(master, caps, on='dc', how='left')
            master['Total_Cap'] = master['Total_Cap'].fillna(0)
        else:
            master['Total_Cap'] = 0
    else:
        master['Total_Cap'] = 0

    # --- Sorting ---
    # Sort by total servers ascending so largest is at the top of the bar chart
    master['total_servers'] = master['active_server_count'] + master['spare_server_count']
    master = master.sort_values(by='total_servers', ascending=True)

    n_bars = len(master)
    if n_bars == 0:
        job.log.warning("No data to plot for servers per DC.")
        return

    # Dynamic height
    fig_height = max(4, n_bars * 0.8)
    
    fig = plt.figure(figsize=(12, fig_height))
    ax = plt.gca()

    y_pos = np.arange(len(master))
    active_vals = master['active_server_count'].to_numpy()
    spare_vals = master['spare_server_count'].to_numpy()

    # Calculate Percentages
    total_region_cpu = master['Total_Cap'].sum()
    dc_labels = []
    
    for dc, cap in zip(master['dc'], master['Total_Cap']):
        pct = (cap / total_region_cpu) * 100 if total_region_cpu > 0 else 0
        dc_labels.append(f"{dc} ({pct:.1f}%)")

    # Plot
    bar_height = 0.6
    p1 = plt.barh(y_pos, active_vals, height=bar_height, color='#00008B', label='Activos')
    p2 = plt.barh(y_pos, spare_vals, left=active_vals, height=bar_height, color='#87CEEB', label='Spare')

    plt.title('Cantidad de Servidores por Datacenter', fontsize=13, fontweight='bold')

    ax.text(-0.4, 1.005, "Datacenter (% de recursos de la Región)", 
            transform=ax.transAxes, fontsize=13, fontweight='bold', 
            va='bottom', ha='left',
            bbox=dict(fc="white", ec="none", alpha=0.0))

    plt.yticks(y_pos, dc_labels)
    plt.grid(axis='x', alpha=0.3)
    plt.legend(loc='lower right')

    max_val = (active_vals + spare_vals).max()
    if max_val > 0:
        plt.xlim(0, max_val * 1.35)

    for i, (act, spr) in enumerate(zip(active_vals, spare_vals)):
        total = act + spr
        if total > 0:
            label_text = f"Activos: {int(act)} / Spare: {int(spr)}"
            plt.text(total + 0.5, i, label_text, va='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    job.log.debug(f"[plot_fig5_servers_per_dc] saved: {out_path}")


def plot_top10_idle_cpu(job, top10_idle_cpu, out_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.DataFrame(top10_idle_cpu or [])
    if df.empty:
        job.log.warning("[plot_top10_idle_cpu] top10_idle_cpu vacío; no se genera figura")
        return

    if "vnf_instance" not in df.columns and "vnf_type" in df.columns:
        df["vnf_instance"] = df["vnf_type"]

    if "flavor:Vcpus" not in df.columns and "resv_cpu" in df.columns:
        df["flavor:Vcpus"] = df["resv_cpu"]

    # Necesitamos cpu_p95_raw numérico
    if "cpu_p95_raw" not in df.columns:
        df["cpu_p95_raw"] = 0

    df["flavor:Vcpus"] = pd.to_numeric(df["flavor:Vcpus"], errors="coerce").fillna(0)
    df["idle_cpu"] = pd.to_numeric(df["idle_cpu"], errors="coerce").fillna(0)
    df["cpu_p95_raw"] = pd.to_numeric(df["cpu_p95_raw"], errors="coerce").fillna(0)

    # Utilizado efectivo = total - idle
    df["Used_eff"] = (df["flavor:Vcpus"] - df["idle_cpu"]).clip(lower=0)

    # Orden
    df = df.sort_values(by="idle_cpu", ascending=False).head(10).copy()

    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 8))

    # 1) Rojo: Utilizado efectivo (base)
    sns.barplot(
        data=df,
        x="Used_eff",
        y="vnf_instance",
        color="#e74c3c",
        label="Utilizado efectivo",
        orient="h"
    )

    # 2) Verde: Ocioso, apilado después del rojo (left=Used_eff)
    plt.barh(
        y=df["vnf_instance"],
        width=df["idle_cpu"],
        left=df["Used_eff"],
        color="#2ecc71",
        label="Ocioso"
    )

    # 3) Azul: Utilizado local (P95 raw), apilado después de rojo+verde
    plt.barh(
        y=df["vnf_instance"],
        width=df["cpu_p95_raw"],
        left=(df["Used_eff"] + df["idle_cpu"]),
        color="#3498db",
        label="Utilizado local"
    )

    # Labels: mantenemos como antes, mostrando idle (verde)
    for i, (used_eff, idle) in enumerate(zip(df["Used_eff"], df["idle_cpu"])):
        total = used_eff + idle
        plt.text(total + 10, i, f"{int(idle)} vCPUs", va="center", fontweight="bold", color="black")

    plt.title("Top 10: Mayor Cantidad de CPU Ociosa", fontsize=14, pad=20)
    plt.xlabel("Capacidad (vCPUs)", fontsize=12)
    plt.ylabel("VNF Instance", fontsize=12)

    # Leyenda con orden: Ocioso / Utilizado efectivo / Utilizado local (o el que prefieras)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = ["Ocioso", "Utilizado efectivo", "Utilizado local"]
    ordered = [(h, l) for (h, l) in zip(handles, labels) if l in order]
    ordered.sort(key=lambda x: order.index(x[1]))
    plt.legend([h for h, _ in ordered], [l for _, l in ordered], loc="lower right", title="")

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    job.log.debug(f"[plot_top10_idle_cpu] saved: {out_path}")

def draw_colored_text(ax, x, y, text_parts, colors, separator=" / ", sep_color="black", **kwargs):
    """
    Draws text with different colors for each part sequentially.
    """
    # Force a draw to ensure renderer is available for bounding box calculations
    try:
        renderer = ax.figure.canvas.get_renderer()
    except Exception:
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
    
    curr_x = x
    inv = ax.transData.inverted()
    
    for i, (part, color) in enumerate(zip(text_parts, colors)):
        # Apply kwarg overrides if needed
        t = ax.text(curr_x, y, part, color=color, **kwargs)
        
        # Calculate width in data coordinates
        bbox = t.get_window_extent(renderer=renderer)
        # Transform corners to data coords
        p0 = inv.transform(bbox.p0)
        p1 = inv.transform(bbox.p1)
        width_data = p1[0] - p0[0]
        
        curr_x += width_data
        
        # Draw Separator if not last
        if i < len(text_parts) - 1:
            sep_t = ax.text(curr_x, y, separator, color=sep_color, **kwargs)
            bbox_sep = sep_t.get_window_extent(renderer=renderer)
            p0_s = inv.transform(bbox_sep.p0)
            p1_s = inv.transform(bbox_sep.p1)
            width_sep = p1_s[0] - p0_s[0]
            curr_x += width_sep

def plot_top10_idle_cpu_updated(job, top10_idle_cpu, out_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    df = pd.DataFrame(top10_idle_cpu or [])
    if df.empty:
        job.log.warning("[plot_top10_idle_cpu] top10_idle_cpu vacío; no se genera figura")
        return

    # Normalization
    if "vnf_instance" not in df.columns and "vnf_type" in df.columns:
        df["vnf_instance"] = df["vnf_type"]

    if "flavor:Vcpus" not in df.columns and "resv_cpu" in df.columns:
        df["flavor:Vcpus"] = df["resv_cpu"]

    if "cpu_p95_raw" not in df.columns:
        df["cpu_p95_raw"] = 0

    df["flavor:Vcpus"] = pd.to_numeric(df["flavor:Vcpus"], errors="coerce").fillna(0)
    df["idle_cpu"] = pd.to_numeric(df["idle_cpu"], errors="coerce").fillna(0)
    df["cpu_p95_raw"] = pd.to_numeric(df["cpu_p95_raw"], errors="coerce").fillna(0)
    df["threshold_high"] = pd.to_numeric(df["threshold_high"], errors="coerce").fillna(70)
    df["threshold_low"] = pd.to_numeric(df["threshold_low"], errors="coerce").fillna(50)


    # Derived
    df["Used_eff"] = (df["flavor:Vcpus"] - df["idle_cpu"]).clip(lower=0)
    
    
    # Filter: Remove items with idle <= 0
    df = df[df["idle_cpu"] > 0]

    # SORT: Descending by IDLE (Libre)
    # Get Top 10
    df = df.sort_values(by="idle_cpu", ascending=False).head(10)
    # Reverse so largest is at the TOP (since barh plots index 0 at bottom)
    df = df.iloc[::-1].copy()

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    # STACK:
    # 1. Local (Red)
    p1 = plt.barh(
        y=df["vnf_instance"],
        width=df["cpu_p95_raw"],
        color="#e74c3c",
        label="Utilizado local"
    )

    # 2. Effective Remainder (Light Red) = Used_eff - Local
    eff_remainder = (df["Used_eff"] - df["cpu_p95_raw"]).clip(lower=0)
    p2 = plt.barh(
        y=df["vnf_instance"],
        width=eff_remainder,
        left=df["cpu_p95_raw"],
        color="#ff7675",
        label="Utilizado efectivo"
    )

    # 3. Ociosa (Green)
    current_left = df["cpu_p95_raw"] + eff_remainder
    p3 = plt.barh(
        y=df["vnf_instance"],
        width=df["idle_cpu"],
        left=current_left,
        color="#2ecc71",
        label="Ociosa"
    )

    # --- Setup Layout BEFORE Drawing Text ---
    plt.title("Top 10: Mayor Cantidad de CPU Ociosa", fontsize=14, pad=20)
    plt.xlabel("Capacidad (vCPUs)", fontsize=12)
    plt.ylabel("VNF Instance", fontsize=12)
    plt.legend(loc="lower right", title="")

    # Set Limits now
    max_val = (current_left + df["idle_cpu"]).max()
    if max_val > 0:
        plt.xlim(0, max_val * 1.55) # Generous padding for labels

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    # Force Draw to ensure transforms match final layout
    fig.canvas.draw()

    # Labels with Colors
    # Blue: #3498db -> Local
    # Red: #e74c3c -> Effective
    # Green: #2ecc71 -> Libre
    
    for i, (local, eff_rem, idle) in enumerate(zip(df["cpu_p95_raw"], eff_remainder, df["idle_cpu"])):
        eff_total = local + eff_rem
    
        parts = [f"Local: {int(local)}", f"Efectiva: {int(eff_total)}", f"Ociosa: {int(idle)}"]
        colors = ["#e74c3c", "#ff7675", "#2ecc71"]
        
        total_len = local + eff_rem + idle
    
        draw_colored_text(
            ax, 
            x=total_len + (total_len * 0.01) + 0.1, 
            y=i, 
            text_parts=parts, 
            colors=colors,
            va="center", fontweight="bold", fontsize=9
        )

        # Threshold Indicators
        # Low
        t_low = df.iloc[i]["threshold_low"]
        limit_low = total_len * (t_low / 100.0)
        ax.vlines(x=limit_low, ymin=i-0.4, ymax=i+0.4, colors='black', linestyles=':', linewidth=1.5, alpha=0.9)
        ax.text(limit_low, i, f"{int(t_low)}%", color="black", fontsize=8, weight="bold", ha="center", va="center", 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))
        
        # High
        t_high = df.iloc[i]["threshold_high"]
        limit_high = total_len * (t_high / 100.0)
        ax.vlines(x=limit_high, ymin=i-0.4, ymax=i+0.4, colors='black', linestyles='--', linewidth=1.5, alpha=0.9)
        ax.text(limit_high, i, f"{int(t_high)}%", color="black", fontsize=8, weight="bold", ha="center", va="center", 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    job.log.debug(f"[plot_top10_idle_cpu] saved: {out_path}")

def plot_top10_idle_mem(job, top10_idle_mem, out_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.DataFrame(top10_idle_mem or [])
    if df.empty:
        job.log.warning("[plot_top10_idle_mem] top10_idle_mem vacío; no se genera figura")
        return

    if "vnf_instance" not in df.columns and "vnf_type" in df.columns:
        df["vnf_instance"] = df["vnf_type"]

    if "memory_alloc_gb" not in df.columns:
        if "flavor:Ram" in df.columns:
            df["memory_alloc_gb"] = df["flavor:Ram"]
        elif "resv_mem" in df.columns:
            df["memory_alloc_gb"] = df["resv_mem"]

    if "idle_mem" not in df.columns:
        df["idle_mem"] = 0

    # Necesitamos mem_p95_raw (bytes) -> GB
    if "mem_p95_raw" not in df.columns:
        df["mem_p95_raw_gb"] = 0
    else:
        df["mem_p95_raw_gb"] = pd.to_numeric(df["mem_p95_raw"], errors="coerce").fillna(0) / (1024 ** 3)

    df["memory_alloc_gb"] = pd.to_numeric(df["memory_alloc_gb"], errors="coerce").fillna(0)
    df["idle_mem"] = pd.to_numeric(df["idle_mem"], errors="coerce").fillna(0)

    df["Used_eff"] = (df["memory_alloc_gb"] - df["idle_mem"]).clip(lower=0)

    # Orden EXACTO como tu original para MEM
    df = df.sort_values(by="idle_mem", ascending=False).iloc[1:11].copy()

    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 8))

    # 1) Rojo: Utilizado efectivo
    sns.barplot(
        data=df,
        x="Used_eff",
        y="vnf_instance",
        color="#e74c3c",
        label="Utilizado efectivo",
        orient="h"
    )

    # 2) Verde: Ocioso
    plt.barh(
        y=df["vnf_instance"],
        width=df["idle_mem"],
        left=df["Used_eff"],
        color="#2ecc71",
        label="Ocioso"
    )

    # 3) Azul: Utilizado local (P95 raw)
    plt.barh(
        y=df["vnf_instance"],
        width=df["mem_p95_raw_gb"],
        left=(df["Used_eff"] + df["idle_mem"]),
        color="#3498db",
        label="Utilizado local"
    )

    for i, (used_eff, idle) in enumerate(zip(df["Used_eff"], df["idle_mem"])):
        total = used_eff + idle
        plt.text(total + 10, i, f"{int(idle)} GB", va="center", fontweight="bold", color="black")

    plt.title("Top 10: Mayor Cantidad de RAM Ociosa", fontsize=14, pad=20)
    plt.xlabel("Capacidad (GB)", fontsize=12)
    plt.ylabel("VNF Instance", fontsize=12)

    handles, labels = plt.gca().get_legend_handles_labels()
    order = ["Ocioso", "Utilizado efectivo", "Utilizado local"]
    ordered = [(h, l) for (h, l) in zip(handles, labels) if l in order]
    ordered.sort(key=lambda x: order.index(x[1]))
    plt.legend([h for h, _ in ordered], [l for _, l in ordered], loc="lower right", title="")

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    job.log.debug(f"[plot_top10_idle_mem] saved: {out_path}")

def plot_top10_idle_mem_updated(job, top10_idle_mem, out_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = pd.DataFrame(top10_idle_mem or [])
    if df.empty:
        job.log.warning("[plot_top10_idle_mem] top10_idle_mem vacío; no se genera figura")
        return

    if "vnf_instance" not in df.columns and "vnf_type" in df.columns:
        df["vnf_instance"] = df["vnf_type"]

    if "memory_alloc_gb" not in df.columns:
        if "flavor:Ram" in df.columns:
            df["memory_alloc_gb"] = df["flavor:Ram"]
        elif "resv_mem" in df.columns:
            df["memory_alloc_gb"] = df["resv_mem"]

    if "idle_mem" not in df.columns:
        df["idle_mem"] = 0

    if "mem_p95_raw" not in df.columns:
        df["mem_p95_raw_gb"] = 0
    else:
        df["mem_p95_raw_gb"] = pd.to_numeric(df["mem_p95_raw"], errors="coerce").fillna(0) / (1024 ** 3)

    df["memory_alloc_gb"] = pd.to_numeric(df["memory_alloc_gb"], errors="coerce").fillna(0)
    df["idle_mem"] = pd.to_numeric(df["idle_mem"], errors="coerce").fillna(0)
    df["threshold_high"] = pd.to_numeric(df["threshold_high"], errors="coerce").fillna(70)
    df["threshold_low"] = pd.to_numeric(df["threshold_low"], errors="coerce").fillna(50)

    
    # Derived
    df["Used_eff"] = (df["memory_alloc_gb"] - df["idle_mem"]).clip(lower=0)
    
    # Sort
    # Filter: Remove items with idle <= 0
    df = df[df["idle_mem"] > 0]
    
    # SORT: Descending by IDLE (Libre)
    df = df.sort_values(by="idle_mem", ascending=False).head(10)
    df = df.iloc[::-1].copy()

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    # STACK
    # 1. Local (Red)
    p1 = plt.barh(
        y=df["vnf_instance"],
        width=df["mem_p95_raw_gb"],
        color="#e74c3c",
        label="Utilizado local"
    )

    # 2. Effective Remainder (Light Red)
    eff_remainder = (df["Used_eff"] - df["mem_p95_raw_gb"]).clip(lower=0)
    p2 = plt.barh(
        y=df["vnf_instance"],
        width=eff_remainder,
        left=df["mem_p95_raw_gb"],
        color="#ff7675",
        label="Utilizado efectivo"
    )

    # 3. Ociosa (Green)
    current_left = df["mem_p95_raw_gb"] + eff_remainder
    p3 = plt.barh(
        y=df["vnf_instance"],
        width=df["idle_mem"],
        left=current_left,
        color="#2ecc71",
        label="Ociosa"
    )

    # --- Setup Layout BEFORE Drawing Text ---
    plt.title("Top 10: Mayor Cantidad de RAM Ociosa", fontsize=14, pad=20)
    plt.xlabel("Capacidad (GB)", fontsize=12)
    plt.ylabel("VNF Instance", fontsize=12)
    plt.legend(loc="lower right", title="")
    
    max_val = (current_left + df["idle_mem"]).max()
    if max_val > 0:
        plt.xlim(0, max_val * 1.55)

    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    fig.canvas.draw() # Finalize coords

    # Labels with Colors
    for i, (local, eff_rem, idle) in enumerate(zip(df["mem_p95_raw_gb"], eff_remainder, df["idle_mem"])):
        eff_total = local + eff_rem
        
        parts = [f"Local: {int(local)} GB", f"Efectiva: {int(eff_total)} GB", f"Ociosa: {int(idle)} GB"]
        colors = ["#e74c3c", "#ff7675", "#2ecc71"]
        
        total_len = local + eff_rem + idle
        
        draw_colored_text(
            ax, 
            x=total_len + (total_len * 0.01) + 0.1, 
            y=i, 
            text_parts=parts, 
            colors=colors,
            va="center", fontweight="bold", fontsize=9
        )

        # Threshold Indicators
        # Low
        t_low = df.iloc[i]["threshold_low"]
        limit_low = total_len * (t_low / 100.0)
        ax.vlines(x=limit_low, ymin=i-0.4, ymax=i+0.4, colors='black', linestyles=':', linewidth=1.5, alpha=0.9)
        ax.text(limit_low, i, f"{int(t_low)}%", color="black", fontsize=8, weight="bold", ha="center", va="center", 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))
        
        # High
        t_high = df.iloc[i]["threshold_high"]
        limit_high = total_len * (t_high / 100.0)
        ax.vlines(x=limit_high, ymin=i-0.4, ymax=i+0.4, colors='black', linestyles='--', linewidth=1.5, alpha=0.9)
        ax.text(limit_high, i, f"{int(t_high)}%", color="black", fontsize=8, weight="bold", ha="center", va="center", 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    job.log.debug(f"[plot_top10_idle_mem] saved: {out_path}")

def plot_top10_idle_disk(job, top10_idle_disk, out_path):
    """
    Replica EXACTAMENTE el gráfico original (seaborn) para Disco:
      - barra verde = total reservado (disk_alloc_gb)
      - barra roja  = Used = total - idle_disk (overlay)
      - seaborn whitegrid + despine
      - labels a la derecha: total + (total*0.01) y "X GB"
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.DataFrame(top10_idle_disk or [])
    if df.empty:
        job.log.warning("[plot_top10_idle_disk] top10_idle_disk vacío; no se genera figura")
        return

    # Compatibilidad nombres
    if "vnf_instance" not in df.columns and "vnf_type" in df.columns:
        df["vnf_instance"] = df["vnf_type"]

    # Total capacity en el script original: disk_alloc_gb
    # En tu data suele venir como flavor:Disk o resv_disk (según cómo lo generaste)
    if "disk_alloc_gb" not in df.columns:
        if "flavor:Disk" in df.columns:
            df["disk_alloc_gb"] = df["flavor:Disk"]
        elif "resv_disk" in df.columns:
            df["disk_alloc_gb"] = df["resv_disk"]

    # Idle
    if "idle_disk" not in df.columns:
        df["idle_disk"] = None

    # Numéricos
    df["disk_alloc_gb"] = pd.to_numeric(df["disk_alloc_gb"], errors="coerce").fillna(0)
    df["idle_disk"] = pd.to_numeric(df["idle_disk"], errors="coerce").fillna(0)

    # Used = total - idle
    df["Used"] = (df["disk_alloc_gb"] - df["idle_disk"]).clip(lower=0)

    # Orden EXACTO como original
    df = df.sort_values(by="idle_disk", ascending=False).head(10).copy()

    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 8))

    sns.barplot(
        data=df,
        x="disk_alloc_gb",
        y="vnf_instance",
        color="#2ecc71",
        label="Ocioso",
        orient="h"
    )

    sns.barplot(
        data=df,
        x="Used",
        y="vnf_instance",
        color="#e74c3c",
        label="Utilizado",
        orient="h"
    )

    for i, (total, idle) in enumerate(zip(df["disk_alloc_gb"], df["idle_disk"])):
        plt.text(total + (total * 0.01), i, f"{int(idle)} GB", va="center", fontweight="bold", color="black")

    plt.title("Top 10: Mayor Cantidad de Disco Ocioso", fontsize=14, pad=20)
    plt.xlabel("Capacidad (GB)", fontsize=12)
    plt.ylabel("VNF Instance", fontsize=12)
    plt.legend(loc="lower right", title="")

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    job.log.debug(f"[plot_top10_idle_disk] saved: {out_path}")

def plot_top10_idle_disk_updated(job, top10_idle_disk, out_path):
    """
    Replica el gráfico original (seaborn) para Disco pero con:
      - Sorting reverso (mayor Idle arriba)
      - Labels: "Utilizado: <val> / Ocioso: <val>"
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = pd.DataFrame(top10_idle_disk or [])
    if df.empty:
        job.log.warning("[plot_top10_idle_disk] top10_idle_disk vacío; no se genera figura")
        return

    # Compatibilidad nombres
    if "vnf_instance" not in df.columns and "vnf_type" in df.columns:
        df["vnf_instance"] = df["vnf_type"]

    # Total capacity
    if "disk_alloc_gb" not in df.columns:
        if "flavor:Disk" in df.columns:
            df["disk_alloc_gb"] = df["flavor:Disk"]
        elif "resv_disk" in df.columns:
            df["disk_alloc_gb"] = df["resv_disk"]

    # Idle
    if "idle_disk" not in df.columns:
        df["idle_disk"] = 0

    # Numéricos
    df["disk_alloc_gb"] = pd.to_numeric(df["disk_alloc_gb"], errors="coerce").fillna(0)
    df["idle_disk"] = pd.to_numeric(df["idle_disk"], errors="coerce").fillna(0)
    df["threshold_high"] = pd.to_numeric(df["threshold_high"], errors="coerce").fillna(70)
    df["threshold_low"] = pd.to_numeric(df["threshold_low"], errors="coerce").fillna(50)


    # Used = total - idle
    df["Used"] = (df["disk_alloc_gb"] - df["idle_disk"]).clip(lower=0)

    # Orden EXACTO como original
    # Filter: Remove items with idle <= 0
    df = df[df["idle_disk"] > 0]

    # SORT: Descending by IDLE (Libre)
    df = df.sort_values(by="idle_disk", ascending=False).head(10)
    # Reverse so largest is at the TOP (consistent with other plots)
    df = df.iloc[::-1].copy()

    sns.set_style("whitegrid")
    # For colored text logic to work similarly (ax.text), we can get ax
    fig, ax = plt.subplots(figsize=(14, 8))

    # STACK:
    # 1. Used (Red)
    p1 = plt.barh(
        y=df["vnf_instance"],
        width=df["Used"],
        color="#e74c3c",
        label="Utilizado"
    )

    # 2. Idle (Green)
    p2 = plt.barh(
        y=df["vnf_instance"],
        width=df["idle_disk"],
        left=df["Used"],
        color="#2ecc71",
        label="Ocioso"
    )

    # Labels: "Utilizado: <val> / Libre: <val>"
    fig.canvas.draw() # Finalize coords for draw_colored_text
    for i, (total, idle, used) in enumerate(zip(df["disk_alloc_gb"], df["idle_disk"], df["Used"])):
        parts = [f" Utilizado: {int(used)} GB          ", f" Libre: {int(idle)} GB"]
        colors = ["#e74c3c", "#2ecc71"]
        
        draw_colored_text(
            ax, 
            x=total + (total * 0.01) + 0.1, 
            y=i, 
            text_parts=parts, 
            colors=colors,
            va="center", fontweight="bold", fontsize=9
        )

        # Threshold Indicators
        # Low
        t_low = df.iloc[i]["threshold_low"]
        limit_low = total * (t_low / 100.0)
        ax.vlines(x=limit_low, ymin=i-0.4, ymax=i+0.4, colors='black', linestyles=':', linewidth=1.5, alpha=0.9)
        ax.text(limit_low, i, f"{int(t_low)}%", color="black", fontsize=8, weight="bold", ha="center", va="center", 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))

        # High
        t_high = df.iloc[i]["threshold_high"]
        limit_high = total * (t_high / 100.0)
        ax.vlines(x=limit_high, ymin=i-0.4, ymax=i+0.4, colors='black', linestyles='--', linewidth=1.5, alpha=0.9)
        ax.text(limit_high, i, f"{int(t_high)}%", color="black", fontsize=8, weight="bold", ha="center", va="center", 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))


    plt.title("Top 10: Mayor Cantidad de Disco Ocioso", fontsize=14, pad=20)
    plt.xlabel("Capacidad (GB)", fontsize=12)
    plt.ylabel("VNF Instance", fontsize=12)
    plt.legend(loc="lower right", title="")

    # Set Limits now
    max_val = (df["disk_alloc_gb"]).max() 
    if max_val > 0:
        plt.xlim(0, max_val * 1.55) # Generous padding for labels

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    job.log.debug(f"[plot_top10_idle_disk] saved: {out_path}")

def plot_vm_matrix_heatmap(job, vms_info, cluster_metrics, out_png_path):
    """
    Replica el heatmap del notebook:
      - pivot cluster x VNF contando VMs
      - agrega clusters faltantes (missing_az) derivados de cluster_metrics.role != 'active'
      - 2 capas: non-zero con cmap YlOrRd y ceros en blanco con anotación 0
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # -------------------------
    # 1) Derivar missing_az desde cluster_metrics
    # -------------------------
    missing_az = set()
    for r in (cluster_metrics or []):
        az = r.get("cluster")
        role = r.get("role")
        if not az or az in ("internal", "Unknown"):
            continue
        if role != "active":
            missing_az.add(az)

    missing_az = sorted(missing_az)
    job.log.debug(f"[plot_vm_matrix_heatmap] derived missing_az={len(missing_az)}")

    # -------------------------
    # 2) Normalizar vms_info a DF mínimo: (cluster, vnf, vm)
    # -------------------------
    rows = []
    for it in (vms_info or []):
        vm_id = it.get("vm")
        cluster = (it.get("hypervisor_vm") or {}).get("cluster")

        vnfc_vm = it.get("vnfc_vm") or {}
        vnfc_instancevnf = vnfc_vm.get("vnfc_instancevnf") or {}
        vnf_instancevnf = vnfc_instancevnf.get("vnf_instancevnf") or {}
        vnf_name = vnf_instancevnf.get("name")

        if not vm_id or not cluster or not vnf_name:
            continue

        if cluster in ("internal", "Unknown"):
            continue

        rows.append({"cluster": str(cluster), "vnf": str(vnf_name), "vm": str(vm_id)})

    df = pd.DataFrame(rows)
    job.log.debug(f"[plot_vm_matrix_heatmap] rows={len(df)}")

    if df.empty:
        raise Exception("plot_vm_matrix_heatmap: df vacío (no hay vm/cluster/vnf para pivotear).")

    # -------------------------
    # 3) Pivot (cluster x vnf) contando VMs
    # -------------------------
    vm_matrix = df.pivot_table(
        index="cluster",
        columns="vnf",
        values="vm",
        aggfunc="count",
        fill_value=0,
    )

    # Agregar missing_az con ceros
    for az in missing_az:
        if az not in vm_matrix.index:
            vm_matrix.loc[az] = 0

    vm_matrix = vm_matrix.sort_index()
    vm_matrix.index.name = "Cluster (AZ)"
    vm_matrix.columns.name = "VNF"

    data = vm_matrix.copy().astype(float)
    zero_mask = data == 0

    # -------------------------
    # 4) Plot 2 capas (igual al notebook)
    # -------------------------
    sns.set_style("whitegrid")

    fig, ax1 = plt.subplots(figsize=(14, 6))

    annot_matrix = data.copy()
    annot_matrix[zero_mask] = np.nan  # no anotar ceros en la capa 1

    # Capa 1: non-zero
    sns.heatmap(
        data.mask(zero_mask),
        annot=annot_matrix,
        fmt=".0f",
        cmap="YlOrRd",
        cbar=True,
        linewidths=0.5,
        linecolor="black",
        ax=ax1,
    )

    # Capa 2: ceros en blanco, anotados
    sns.heatmap(
        data.where(zero_mask),
        annot=data.where(zero_mask),
        fmt=".0f",
        cmap=sns.color_palette(["#ffffff"], as_cmap=True),
        cbar=False,
        linewidths=0.5,
        linecolor="black",
        ax=ax1,
    )

    ax1.set_title("Matriz Cluster VNF")
    ax1.set_xlabel("VNF")
    ax1.set_ylabel("Cluster (AZ)")

    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    job.log.debug(f"[plot_vm_matrix_heatmap] saved: {out_png_path}")


def plot_vm_matrix_heatmap_by_dc(job, vms_info, dc_metrics, out_png_path):
    """
    Heatmap DC x VNF counting VMs.
    Similar to plot_vm_matrix_heatmap but grouped by datacenter instead of cluster.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # -------------------------
    # 1) Get all DCs from dc_metrics (to include DCs with 0 VMs)
    # -------------------------
    all_dcs = set()
    for r in (dc_metrics or []):
        dc_dict = r.get("datacenter")
        if isinstance(dc_dict, dict):
            dc_name = dc_dict.get("name")
        else:
            dc_name = r.get("datacenter")
        if dc_name:
            all_dcs.add(dc_name)

    job.log.debug(f"[plot_vm_matrix_heatmap_by_dc] all_dcs={len(all_dcs)}")

    # -------------------------
    # 2) Normalize vms_info to DF: (dc, vnf, vm)
    # -------------------------
    rows = []
    for it in (vms_info or []):
        vm_id = it.get("vm")

        vnfc_vm = it.get("vnfc_vm") or {}
        vnfc_instancevnf = vnfc_vm.get("vnfc_instancevnf") or {}
        
        # Extract DC from vnfc_instancevnf.datacenter
        dc = vnfc_instancevnf.get("datacenter").replace("_", " ")
        
        # Extract VNF name
        vnf_instancevnf = vnfc_instancevnf.get("vnf_instancevnf") or {}
        vnf_name = vnf_instancevnf.get("name")

        if not vm_id or not dc or not vnf_name:
            continue

        rows.append({"dc": str(dc), "vnf": str(vnf_name), "vm": str(vm_id)})

    df = pd.DataFrame(rows)
    job.log.debug(f"[plot_vm_matrix_heatmap_by_dc] rows={len(df)}")

    if df.empty:
        job.log.warning("plot_vm_matrix_heatmap_by_dc: df vacío (no hay vm/dc/vnf para pivotear).")
        return

    # -------------------------
    # 3) Pivot (dc x vnf) counting VMs
    # -------------------------
    vm_matrix = df.pivot_table(
        index="dc",
        columns="vnf",
        values="vm",
        aggfunc="count",
        fill_value=0,
    )

    # Add missing DCs with zeros
    for dc in all_dcs:
        if dc not in vm_matrix.index:
            vm_matrix.loc[dc] = 0

    vm_matrix = vm_matrix.sort_index()
    vm_matrix.index.name = "Datacenter"
    vm_matrix.columns.name = "VNF"

    data = vm_matrix.copy().astype(float)
    zero_mask = data == 0

    # -------------------------
    # 4) Plot 2 layers (same as original)
    # -------------------------
    sns.set_style("whitegrid")

    # Dynamic figure size based on number of DCs and VNFs
    n_dcs = len(vm_matrix.index)
    n_vnfs = len(vm_matrix.columns)
    fig_width = max(14, n_vnfs * 0.8)
    fig_height = max(6, n_dcs * 0.6)

    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))

    annot_matrix = data.copy()
    annot_matrix[zero_mask] = np.nan  # don't annotate zeros in layer 1

    # Layer 1: non-zero
    sns.heatmap(
        data.mask(zero_mask),
        annot=annot_matrix,
        fmt=".0f",
        cmap="YlOrRd",
        cbar=True,
        linewidths=0.5,
        linecolor="black",
        ax=ax1,
    )

    # Layer 2: zeros in white, annotated
    sns.heatmap(
        data.where(zero_mask),
        annot=data.where(zero_mask),
        fmt=".0f",
        cmap=sns.color_palette(["#ffffff"], as_cmap=True),
        cbar=False,
        linewidths=0.5,
        linecolor="black",
        ax=ax1,
    )

    ax1.set_title("Matriz Datacenter x VNF (Cantidad de VMs)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("VNF", fontsize=12)
    ax1.set_ylabel("Datacenter", fontsize=12)

    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    job.log.debug(f"[plot_vm_matrix_heatmap_by_dc] saved: {out_png_path}")


def build_capacity_reserve_df(job, dc_metrics_dict, site_name):
    """
    Entrada:
      - dc_metrics_dict: dict con keys tipo efect_resv_cpu_pct, efect_util_cpu, efect_resv_mem_pct, efect_util_mem,
                        efect_resv_disk_pct, efect_util_disk_pct (y opcionalmente otras)
      - site_name: string (ej "Mayo")
    Salida:
      - pd.DataFrame con columnas: DC, Region (si está), efect_resv_pct, efect_util_pct, type
        (3 filas: CPU, RAM, STORAGE)
    """
    import pandas as pd

    if not isinstance(dc_metrics_dict, dict):
        raise Exception("dc_metrics_dict debe ser dict")

    region = dc_metrics_dict.get("region")

    rows = [
        {
            "DC": f"CPU ({site_name})",
            "Region": region,
            "efect_resv_pct": dc_metrics_dict.get("efect_resv_cpu_pct"),
            "efect_util_pct": dc_metrics_dict.get("efect_util_cpu"),
            "type": "CPU",
        },
        {
            "DC": f"RAM ({site_name})",
            "Region": region,
            "efect_resv_pct": dc_metrics_dict.get("efect_resv_mem_pct"),
            "efect_util_pct": dc_metrics_dict.get("efect_util_mem"),
            "type": "RAM",
        },
        {
            "DC": f"STORAGE ({site_name})",
            "Region": region,
            "efect_resv_pct": dc_metrics_dict.get("efect_resv_disk_pct"),
            "efect_util_pct": dc_metrics_dict.get("efect_util_disk_pct"),
            "type": "STORAGE",
        },
    ]

    df = pd.DataFrame(rows)

    # log mínimo para debug
    job.log.debug(f"[build_capacity_reserve_df] rows={len(df)} cols={list(df.columns)}")
    return df


def build_capacity_reserve_df_multi_dc(job, dc_metrics_list):
    """
    Build capacity reserve DataFrame for multiple DCs.
    
    Entrada:
      - dc_metrics_list: list of dicts, each with keys:
          - datacenter.name (or datacenter dict with 'name')
          - efect_resv_cpu_pct, efect_util_cpu
          - efect_resv_mem_pct, efect_util_mem
          - efect_resv_disk_pct, efect_util_disk_pct
    Salida:
      - pd.DataFrame con columnas: DC, site_name, efect_resv_pct, efect_util_pct, type
        (3 filas per DC: CPU, RAM, STORAGE)
    """
    import pandas as pd

    if not dc_metrics_list:
        return pd.DataFrame(columns=['DC', 'site_name', 'efect_resv_pct', 'efect_util_pct', 'type'])

    rows = []
    for dc_metrics in dc_metrics_list:
        # Extract site name from datacenter dict or direct name
        if isinstance(dc_metrics.get("datacenter"), dict):
            site_name = dc_metrics["datacenter"].get("name", "Unknown")
        else:
            site_name = dc_metrics.get("datacenter", "Unknown")
        
        rows.append({
            "DC": f"CPU ({site_name})",
            "site_name": site_name,
            "efect_resv_pct": dc_metrics.get("efect_resv_cpu_pct"),
            "efect_util_pct": dc_metrics.get("efect_util_cpu"),
            "type": "CPU",
        })
        rows.append({
            "DC": f"RAM ({site_name})",
            "site_name": site_name,
            "efect_resv_pct": dc_metrics.get("efect_resv_mem_pct"),
            "efect_util_pct": dc_metrics.get("efect_util_mem"),
            "type": "RAM",
        })
        rows.append({
            "DC": f"STORAGE ({site_name})",
            "site_name": site_name,
            "efect_resv_pct": dc_metrics.get("efect_resv_disk_pct"),
            "efect_util_pct": dc_metrics.get("efect_util_disk_pct"),
            "type": "STORAGE",
        })

    df = pd.DataFrame(rows)
    job.log.debug(f"[build_capacity_reserve_df_multi_dc] rows={len(df)} DCs={len(dc_metrics_list)}")
    return df

def plot_capacity_reserve(
    job,
    df,
    x_col,
    y_col,
    label_col,
    title,
    save_path,
    color_by="cluster",
    show_labels=True,
    x_label="Effective Utilization (%)",
    y_label="Effective Reserve (%)",
    cmap_name="Dark2",
    new_threshold = None
):
    """
    Versión MAT-clean de plot_capacity_reserve:
      - Sin plt.show()
      - Guarda en save_path
      - Logs con job.log
      - Imports adentro
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    if df is None or getattr(df, "empty", True):
        job.log.warning("[plot_capacity_reserve_mat] df vacío, no se genera figura")
        return

    # Select data
    x = df[x_col] if x_col in df.columns else None
    y = df[y_col] if y_col in df.columns else None
    labels = df[label_col] if label_col in df.columns else None
    groups = df[color_by] if color_by in df.columns else None

    if x is None or y is None or labels is None or groups is None:
        raise Exception(
            f"[plot_capacity_reserve_mat] faltan columnas. "
            f"Necesito: {x_col}, {y_col}, {label_col}, {color_by}. "
            f"Cols={list(df.columns)}"
        )

    # Filter rows with valid X/Y
    mask = x.notna() & y.notna()
    x = x[mask].astype(float).values
    y = y[mask].astype(float).values
    labels = labels[mask].astype(str).values.tolist()
    groups = groups[mask].astype(str).values

    # Shorten labels (remove FQDN endings)
    labels = [val.split(".")[0] for val in labels]

    # Unique colors per group
    unique_groups = sorted(set(groups))
    cmap = plt.cm.get_cmap(cmap_name, max(len(unique_groups), 9))
    group_to_color = {c: cmap(i + 1) for i, c in enumerate(unique_groups)}
    colors = [group_to_color[c] for c in groups]

    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # ---------------------------------------------------------
    # 2D background colormap (reglas por zonas) - estética nueva
    # ---------------------------------------------------------
    low_cut = 20
    high_cut = 80

    n = 400
    xs = np.linspace(0, 100, n)
    ys = np.linspace(0, 100, n)
    X, Y = np.meshgrid(xs, ys)

    if new_threshold:
        
        red_mask = (
            (X < 30) | (X >= 85) | (Y < 30) | (Y >= 85)
        )
        green_mask = (
            (X >= 70) & (X <= 85) & (Y >= 70) & (Y <= 85)
        )
    
    else:
        red_mask = (X < low_cut) | (X >= high_cut) | (Y < low_cut) | (Y >= high_cut)
        green_mask = (X >= 50) & (X <= 70) & (Y >= 50) & (Y <= 70)

    color_idx = np.full(X.shape, 1, dtype=int)  # 1=orange
    color_idx[red_mask] = 0                     # 0=red
    color_idx[green_mask] = 2                   # 2=green

    zone_colors = ["#d62728", "#ff7f0e", "#2ca02c"]  # red, orange, green
    square_cmap = ListedColormap(zone_colors)

    ax.imshow(
        color_idx,
        extent=[0, 100, 0, 100],
        origin="lower",
        cmap=square_cmap,
        alpha=0.22,
        zorder=-2,
        interpolation="nearest",
    )

    # Textos de bandas (como tu versión nueva)
    plt.text(
        7.5, 50, "Baja\nUtilización",
        ha="center", va="center", fontsize=12, weight="bold",
        color="gray", alpha=0.8, zorder=1
    )
    plt.text(
        50, 92.5, "Alta\nReserva",
        ha="center", va="top", fontsize=12, weight="bold",
        color="darkorange", alpha=0.9, zorder=1
    )
    plt.text(
        50, 7.5, "Baja\nReserva",
        ha="center", va="center", fontsize=12, weight="bold",
        color="gray", alpha=0.8, zorder=1
    )
    plt.text(
        98, 50, "Alta\nUtilización",
        ha="right", va="center", fontsize=12, weight="bold",
        color="darkorange", alpha=0.9, zorder=1
    )

    # Líneas centrales
    plt.axhline(y=50, color="gray", linestyle="-", alpha=0.5)
    plt.axvline(x=50, color="gray", linestyle="-", alpha=0.5)

    # Scatter
    plt.scatter(x, y, s=120, c=colors, edgecolors="white", alpha=0.85, zorder=3)

    if show_labels:
        for xi, yi, txt in zip(x, y, labels):
            plt.annotate(
                txt,
                (xi, yi),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                alpha=0.8,
                zorder=4,
            )

    # Textos de cuadrantes
    plt.text(
        25, 75, "Oportunidades de\nOptimizacion",
        ha="center", va="center", fontsize=14, color="gray", alpha=0.5, weight="bold",
    )
    plt.text(
        75, 75, "Uso eficiente de Reserva\n/ Capacidad de Crecimiento Limitado",
        ha="center", va="center", fontsize=14, color="gray", alpha=0.5, weight="bold",
    )
    plt.text(
        25, 25, "Disponibiliad de \nInfrastuctura",
        ha="center", va="center", fontsize=14, color="gray", alpha=0.5, weight="bold",
    )
    plt.text(
        75, 25, "Oportunidad de mejoras\nOperacionales",
        ha="center", va="center", fontsize=14, color="gray", alpha=0.5, weight="bold",
    )

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel(x_label, fontsize=12, weight="bold")
    plt.ylabel(y_label, fontsize=12, weight="bold")
    plt.title(title, fontsize=16, weight="bold", pad=20)
    plt.grid(True, linestyle=":", alpha=0.3)

    # Legend
    handles = [
        plt.Line2D([], [], color=group_to_color[c], marker="o", linestyle="", label=c)
        for c in unique_groups
    ]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    job.log.info(f"[plot_capacity_reserve_mat] saved: {save_path}")



def plot_effective_util_vs_reserve_by_cluster(job, cluster_metrics, site, figs_dir):
    """
    Genera 2 scatter plots (CPU y RAM) usando cluster_metrics "a secas",
    con la misma estética (fondo por zonas) que plot_capacity_reserve_mat.

    cluster_metrics: list[dict] (cada dict tiene cluster, role, efect_util_*_pct, efect_resv_*_pct, etc.)
    Guarda:
      - <site>_cpu_eff_util_vs_reserve.png
      - <site>_mem_eff_util_vs_reserve.png
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    df = pd.DataFrame(cluster_metrics or [])
    if df.empty:
        job.log.warning("[plot_eff_util_vs_reserve_by_cluster] cluster_metrics vacío; no se generan gráficos")
        return {"cpu_path": None, "mem_path": None}

    for c in ["cluster", "role"]:
        if c not in df.columns:
            raise Exception(f"[plot_eff_util_vs_reserve_by_cluster] falta columna requerida: {c}")

    # Solo active
    df = df[df["role"].astype(str).str.lower().eq("active")].copy()
    if df.empty:
        job.log.warning("[plot_eff_util_vs_reserve_by_cluster] no hay clusters role=active; no se generan gráficos")
        return {"cpu_path": None, "mem_path": None}

    def _plot(df_in, x_col, y_col, title, x_label, y_label, out_path, cmap_name="nipy_spectral",new_threshold=None):
        # Validar columnas
        for c in ["cluster", x_col, y_col]:
            if c not in df_in.columns:
                raise Exception(f"[plot_eff_util_vs_reserve_by_cluster] falta columna requerida: {c}")

        dfp = df_in[["cluster", x_col, y_col]].copy()
        dfp[x_col] = pd.to_numeric(dfp[x_col], errors="coerce")
        dfp[y_col] = pd.to_numeric(dfp[y_col], errors="coerce")
        dfp = dfp.dropna(subset=[x_col, y_col]).copy()

        if dfp.empty:
            job.log.warning(f"[plot_eff_util_vs_reserve_by_cluster] sin datos válidos para {title}; no se genera gráfico")
            return

        x = dfp[x_col].apply(lambda x: 100 if x> 100 else x).astype(float).values
        y = dfp[y_col].astype(float).values
        clusters = dfp["cluster"].astype(str).values

        unique_clusters = sorted(set(clusters))
        cmap = plt.cm.get_cmap(cmap_name, max(len(unique_clusters), 9))
        cluster_to_color = {c: cmap(i + 1) for i, c in enumerate(unique_clusters)}
        colors = [cluster_to_color[c] for c in clusters]

        plt.figure(figsize=(12, 10))
        ax = plt.gca()

        # ---- Fondo por zonas (misma estética nueva) ----
        low_cut = 20
        high_cut = 80

        n = 400
        xs = np.linspace(0, 100, n)
        ys = np.linspace(0, 100, n)
        X, Y = np.meshgrid(xs, ys)
        
        if new_threshold:
            red_mask = (
                (X < 30) | (X >= 85) | (Y < 30) | (Y >= 85)
            )
            green_mask = (
                (X >= 70) & (X <= 85) & (Y >= 70) & (Y <= 85)
            )
        else:
            red_mask = (X < low_cut) | (X >= high_cut) | (Y < low_cut) | (Y >= high_cut)
            green_mask = (X >= 50) & (X <= 70) & (Y >= 50) & (Y <= 70)

        color_idx = np.full(X.shape, 1, dtype=int)  # orange
        color_idx[red_mask] = 0                     # red
        color_idx[green_mask] = 2                   # green

        zone_colors = ["#d62728", "#ff7f0e", "#2ca02c"]
        square_cmap = ListedColormap(zone_colors)

        ax.imshow(
            color_idx,
            extent=[0, 100, 0, 100],
            origin="lower",
            cmap=square_cmap,
            alpha=0.22,
            zorder=-2,
            interpolation="nearest",
        )

        # Textos “bandas” (sin axvspan/axhspan)
        plt.text(7.5, 50, "Baja\nUtilización", ha="center", va="center",
                 fontsize=12, weight="bold", color="gray", alpha=0.8, zorder=1)
        plt.text(50, 92.5, "Alta\nReserva", ha="center", va="top",
                 fontsize=12, weight="bold", color="darkorange", alpha=0.9, zorder=1)
        plt.text(50, 7.5, "Baja\nReserva", ha="center", va="center",
                 fontsize=12, weight="bold", color="gray", alpha=0.8, zorder=1)
        plt.text(98, 50, "Alta\nUtilización", ha="right", va="center",
                 fontsize=12, weight="bold", color="darkorange", alpha=0.9, zorder=1)

        # Líneas centro
        plt.axhline(y=50, color="gray", linestyle="-", alpha=0.5)
        plt.axvline(x=50, color="gray", linestyle="-", alpha=0.5)

        # Scatter
        plt.scatter(x, y, s=120, c=colors, edgecolors="white", alpha=0.85, zorder=3)

        # show_labels=False (como pediste)
        # (no anotamos clusters)

        # Textos cuadrantes
        plt.text(25, 75, "Oportunidades de\nOptimizacion",
                 ha="center", va="center", fontsize=14, color="gray", alpha=0.5, weight="bold")
        plt.text(75, 75, "Uso eficiente de Reserva\n/ Capacidad de Crecimiento Limitado",
                 ha="center", va="center", fontsize=14, color="gray", alpha=0.5, weight="bold")
        plt.text(25, 25, "Disponibiliad de \nInfrastuctura",
                 ha="center", va="center", fontsize=14, color="gray", alpha=0.5, weight="bold")
        plt.text(75, 25, "Oportunidad de mejoras\nOperacionales",
                 ha="center", va="center", fontsize=14, color="gray", alpha=0.5, weight="bold")

        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.xlabel(x_label, fontsize=12, weight="bold")
        plt.ylabel(y_label, fontsize=12, weight="bold")
        plt.title(title, fontsize=16, weight="bold", pad=20)
        plt.grid(True, linestyle=":", alpha=0.3)

        handles = [
            plt.Line2D([], [], color=cluster_to_color[c], marker="o", linestyle="", label=c)
            for c in unique_clusters
        ]
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        job.log.info(f"[plot_eff_util_vs_reserve_by_cluster] saved: {out_path}")

    cpu_path = os.path.join(figs_dir, f"{site}_cpu_eff_util_vs_reserve.png")
    mem_path = os.path.join(figs_dir, f"{site}_mem_eff_util_vs_reserve.png")

    _plot(
        df,
        x_col="efect_util_cpu_pct",
        y_col="efect_resv_cpu_pct",
        title="Utilizacion Efectiva vs Reserva Efectiva",
        x_label="Utilizacion Efectiva CPU(%)",
        y_label="Reserva Efectiva CPU(%)",
        out_path=cpu_path,
        cmap_name="nipy_spectral",
    )

    _plot(
        df,
        x_col="efect_util_mem_pct",
        y_col="efect_resv_mem_pct",
        title="Utilizacion Efectiva vs Reserva Efectiva",
        x_label="Utilizacion Efectiva RAM(%)",
        y_label="Reserva Efectiva RAM(%)",
        out_path=mem_path,
        cmap_name="nipy_spectral",
        new_threshold=True
    )

    return {"cpu_path": cpu_path, "mem_path": mem_path}

    
MAX_POINTS = 1000000
LIMIT_QUERY = 1000

GET_SERVER_METRICS = """query server_metric_site($sites: [String!], $u_date: timestamptz, $l_date: timestamptz, $limit: Int, $offset: Int) {
  Metrics_Server_Metrics(
    where: {_and: [{time: {_lte: $u_date}},
      {time: {_gte: $l_date}},
      {server:{rack_server:{dc:{_in:$sites}}}}
    ]}
    limit: $limit
    offset: $offset
  ) {
    time
    cluster
    cpu_avg
    cpu_max
    cpu_min
    cpu_p95
    region
    server_id
    site
  } 
}"""

def safe_get(obj, *keys):
    """Safely get nested dictionary value, returning None if any key is missing or None."""
    try:
        for key in keys:
            if obj is None:
                return None
            obj = obj[key]
        return obj
    except (KeyError, TypeError):
        return None


def get_server_metrics(job, mat_client, sites, days=14):
    
    import pandas as pd
    from datetime import datetime, timedelta

    query = GET_SERVER_METRICS
    result = run_paginated_query(
        job,
        mat_client,
        query,
        variables={
            "sites": sites,
            "u_date": datetime.now().isoformat(),
            "l_date": (datetime.now() - timedelta(days=days)).isoformat(),
        },
    )

    server_metrics = result.get("data", {}).get("Metrics_Server_Metrics", [])
    server_metrics_df = pd.DataFrame(server_metrics)
    server_metrics_df_grouped = server_metrics_df.groupby(["server_id","cluster"]).agg({
        "cpu_avg": "mean",
        "cpu_max": "max",
        "cpu_min": "min",
        "cpu_p95": "max",
        "region": "first",
        "site": "first",
    }).reset_index()
    return server_metrics_df_grouped

def run_paginated_query(job, mat_client, query, variables, max_points=MAX_POINTS):
    """
    Ejecuta una query GraphQL con paginación (limit/offset) y acumula resultados.

    - MAT-clean: sin prints, usa job.log.*
    - Reusa el mat_client que te pasan (no crea uno nuevo)
    - Imports adentro (MAT)
    """
    all_data = []
    offset = 0
    total_points = 0
    paginated_vars = (variables or {}).copy()

    data_key = None
    last_result = None

    while True:
        paginated_vars["limit"] = LIMIT_QUERY
        paginated_vars["offset"] = offset

        try:
            last_result = mat_client.graphQL.execute(operation=query, variables=paginated_vars)
        except Exception as e:
            job.log.error(f"[run_paginated_query] GraphQL execute failed at offset={offset}: {e}")
            break

        if not last_result or not isinstance(last_result, dict) or not last_result.get("data"):
            job.log.warning(f"[run_paginated_query] No data returned at offset={offset}. Stopping.")
            break

        if last_result.get("errors"):
            job.log.error(f"[run_paginated_query] Errors in query at offset={offset}: {last_result['errors']}")
            break

        # Tomamos la primera key de "data" como dataset principal
        if data_key is None:
            keys = list(last_result["data"].keys())
            if not keys:
                job.log.warning(f"[run_paginated_query] Empty data keys at offset={offset}. Stopping.")
                break
            data_key = keys[0]

        page_data = last_result["data"].get(data_key)
        if not page_data:
            job.log.info(f"[run_paginated_query] No more rows at offset={offset}. Done.")
            break

        all_data.extend(page_data)
        total_points += len(page_data)

        # Última página si devuelve menos que el límite
        if len(page_data) < LIMIT_QUERY:
            job.log.info(f"[run_paginated_query] Last page: rows={len(page_data)} offset={offset}. Done.")
            break

        if total_points >= max_points:
            job.log.warning(f"[run_paginated_query] Reached max_points={max_points}. Stopping. total_points={total_points}")
            break

        offset += LIMIT_QUERY

    job.log.info(
        f"[run_paginated_query] Finished. total_points={total_points} pages={(offset // LIMIT_QUERY) + (1 if total_points else 0)} limit={LIMIT_QUERY}"
    )
    job.log.debug(f"[run_paginated_query] query={query}")
    job.log.debug(f"[run_paginated_query] variables(base)={variables}")

    if all_data and data_key:
        result_copy = (last_result or {}).copy()
        result_copy["data"] = {data_key: all_data}
        return result_copy

    return {"data": {data_key: []}} if data_key else {"data": {}}

def plot_cluster_cpu_usage(job, server_metrics_df, save_path):
    import matplotlib.pyplot as plt
    import pandas as pd

    df = server_metrics_df.copy()

    clusters = df["cluster"]
    unique_clusters = clusters.unique()
    x_map = {c: i for i, c in enumerate(unique_clusters)}
    x_base = clusters.map(x_map).astype(float).values

    jitter = 0.15
    x_avg = x_base - jitter
    x_p95 = x_base
    x_max = x_base + jitter

    y_avg = df["cpu_avg"]
    y_p95 = df["cpu_p95"]
    y_max = df["cpu_max"]

    plt.figure(figsize=(14, 7))

    band_alpha = 0.19
    plt.axhspan(0, 20, color="gray", alpha=band_alpha, zorder=0)
    plt.axhspan(80, 100, color="red", alpha=band_alpha, zorder=0)

    plt.scatter(x_avg, y_avg, color="green", alpha=0.7, label="Promedio")
    plt.scatter(x_p95, y_p95, color="blue", alpha=0.7, label="P95")
    plt.scatter(x_max, y_max, color="red", alpha=0.7, label="Máximo")

    labels = pd.Series(unique_clusters).apply(
        lambda x: x.split("-")[-1]
    )

    plt.xticks(range(len(unique_clusters)), labels, rotation=75, fontsize=12)
    plt.ylabel("Uso de CPU (%)")
    plt.title("Uso de CPU promedio, P95 y máximo por hipervisor, agrupado por cluster", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    job.log.info(f"[plot_cluster_cpu_usage_mat] saved: {save_path}")



def plot_fig3_normalized_by_dc(job, res_data, out_path, dc_monthly_metrics=None):
    """
    Plot normalized utilization chart with one row per datacenter (DC).
    Aggregates cluster data by DC before plotting.
    Uses DC-level utilization metrics from dc_monthly_metrics if provided.
    
    Parameters:
        job: Job object with logging
        res_data: Resource data dict with vCPU, Memoria, Disco
        out_path: Output file path
        dc_monthly_metrics: Optional list of DC monthly metrics with utilization data
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Default hardcoded DC monthly metrics (will be replaced by query later)
    DEFAULT_DC_MONTHLY_METRICS = [
        {
            "datacenter": {"name": "Nextengo_Nacional"},
            "efect_util_cpu": 39.33746099824241,
            "cpu_p95_pct": 27.638329571104553,
            "efect_util_mem": 63.28664727573506,
            "mem_p95_pct": 54.87988024148328,
            "efect_util_disk_pct": 78.33976011854811
        },
        {
            "datacenter": {"name": "Nextengo_Regional"},
            "efect_util_cpu": 65.54990651003419,
            "cpu_p95_pct": 42.117613803231,
            "efect_util_mem": 62.61284910833529,
            "mem_p95_pct": 46.56315328387277,
            "efect_util_disk_pct": 61.53008601920698
        },
        {
            "datacenter": {"name": "San_Juan"},
            "efect_util_cpu": 64.12572354034421,
            "cpu_p95_pct": 43.70063087688506,
            "efect_util_mem": 61.33024209497921,
            "mem_p95_pct": 45.65674488575821,
            "efect_util_disk_pct": 67.5312008494972
        },
        {
            "datacenter": {"name": "Popotla"},
            "efect_util_cpu": 31.856501328655856,
            "cpu_p95_pct": 20.216584289965557,
            "efect_util_mem": 17.641344527003483,
            "mem_p95_pct": 14.233405249459402,
            "efect_util_disk_pct": 53.749166912201865
        }
    ]

    # Use provided data or fall back to default
    metrics_to_use = dc_monthly_metrics if dc_monthly_metrics else DEFAULT_DC_MONTHLY_METRICS

    # Aggregate by DC
    res_data_dc = aggregate_res_data_by_dc(res_data)
    
    # Build DC utilization lookup from dc_monthly_metrics
    # Keys: dc_name -> {efect_util_cpu, cpu_p95_pct, efect_util_mem, mem_p95_pct, efect_util_disk_pct}
    dc_util_lookup = {}
    for m in metrics_to_use:
        dc_dict = m.get("datacenter")
        if isinstance(dc_dict, dict):
            dc_name = dc_dict.get("name")
        else:
            dc_name = m.get("datacenter")
        if dc_name:
            dc_util_lookup[dc_name] = {
                "efect_util_cpu": m.get("efect_util_cpu"),
                "cpu_p95_pct": m.get("cpu_p95_pct"),
                "efect_util_mem": m.get("efect_util_mem"),
                "mem_p95_pct": m.get("mem_p95_pct"),
                "efect_util_disk_pct": m.get("efect_util_disk_pct"),
            }

    resources = ["vCPU", "Memoria", "Disco"]
    order = _get_dc_order(res_data_dc)

    # Colores (igual notebook)
    c_res = "#e67e22"
    c_spare = "#87CEEB"
    c_free = "#2ecc71"
    c_util = "#ff7675"
    c_p95 = "#e74c3c"

    fig, axes = plt.subplots(1, 3, figsize=(24, max(6, len(order) * 0.8)))
    handles3, labels3 = None, None

    # Region totals (para % en label DC)
    region_totals = {res: float(res_data_dc[res]["df"]["Total_Cap"].sum()) for res in resources}

    for i, res in enumerate(resources):
        df = res_data_dc[res]["df"].set_index("dc").reindex(order).reset_index()
        ax = axes[i]

        idx = np.arange(len(df))
        w = 0.25

        # V19 Fix: Threshold 0.01
        is_spare_dc = (df["v_res"] < 0.01) & (df["v_spare"] > 0)

        prov_cap = np.where(
            is_spare_dc,
            df["v_spare"],
            (df["v_res"] + df["v_free"]),
        ).clip(min=1)

        # Barra 1 - componentes estructura
        p_res = np.where(is_spare_dc, 0, (df["v_res"] / prov_cap) * 100.0)
        p_free = np.where(is_spare_dc, 0, (df["v_free"] / prov_cap) * 100.0)
        p_spare = np.where(is_spare_dc, 100.0, 0.0)

        # Barra 2 - Utilización efectiva (from DC monthly metrics)
        # Barra 3 - P95 (from DC monthly metrics, solo CPU y Mem)
        p_util = np.zeros(len(df))
        p_p95 = np.zeros(len(df))
        
        for j, dc_name in enumerate(df["dc"]):
            util_data = dc_util_lookup.get(dc_name, {})
            if res == "vCPU":
                p_util[j] = util_data.get("efect_util_cpu", 0)
                p_p95[j] = util_data.get("cpu_p95_pct", 0)
            elif res == "Memoria":
                p_util[j] = util_data.get("efect_util_mem", 0)
                p_p95[j] = util_data.get("mem_p95_pct", 0)
            else:  # Disco
                p_util[j] = util_data.get("efect_util_disk_pct", 0)

        # Ploteo Barra 1 (idx-w): Res + Spare + Free
        l = np.zeros(len(df))
        ax.barh(idx - w, p_res, w, left=l, color=c_res, label="Reservado")
        l += p_res
        ax.barh(idx - w, p_spare, w, left=l, color=c_spare, label="Spare")
        l += p_spare
        ax.barh(idx - w, p_free, w, left=l, color=c_free, label="Libre")

        # Escalar utilización relativa a la reserva (p_res)
        # Si util=100%, la barra debe igualar p_res
        p_util_scaled = p_util * p_res / 100.0
        p_p95_scaled = p_p95 * p_res / 100.0

        # Barra 2 (idx): Utilización / Utilización Efectiva
        if res == "Disco":
            label_util = "Utilización Local"
            color_util = c_p95
        else:
            label_util = "Utilización Efectiva"
            color_util = c_util

        ax.barh(idx, p_util_scaled, w, color=color_util, label=label_util)

        # Barra 3 (idx+w): P95 (solo para vCPU y Memoria)
        if res != "Disco":
            ax.barh(idx + w, p_p95_scaled, w, color=c_p95, label="Utilización Local")

        ax.set_xlabel("Porcentaje de Capacidad Total Utilizable (%)", fontsize=14)
        ax.set_yticks(idx)

        # Labels DC solo en el panel izquierdo, con % region
        if i == 0:
            region_total = region_totals[res]
            dc_labels = []
            for _, row in df.iterrows():
                dc_name = row["dc"]
                dc_pct = (float(row["Total_Cap"]) / region_total * 100.0) if region_total > 0 else 0.0
                dc_labels.append(f"{dc_name} ({dc_pct:.1f}%)")
            ax.set_yticklabels(dc_labels, fontsize=14)
            ax.text(-0.3, 1.005, "Datacenter (% de recursos de la Región)", 
                transform=ax.transAxes, 
                fontsize=13, fontweight='bold', color='black',
                ha='center', va='bottom',
                bbox=dict(fc="white", ec="none", alpha=0.0))
        else:
            ax.set_yticklabels([])

        ax.set_title(f"Porcentaje de Utilización - {res}", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        if i == 0:
            handles3, labels3 = ax.get_legend_handles_labels()

    fig.legend(handles3, labels3, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.05), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    job.log.debug(f"[plot_fig3_normalized_by_dc] saved: {out_path}")

