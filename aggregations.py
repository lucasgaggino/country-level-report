import pandas as pd

def create_overview_info(cluster_metrics, server_info, vnf_types, vm_info):
    df_servers = pd.DataFrame(server_info)
    df_servers = df_servers[~df_servers["hostname"].str.contains("hds", case=False, na=False)]
    df_servers["dc"] = df_servers["rack_server"].apply(lambda x: x["dc_rack"]["name"])
    
    df_clusters = pd.DataFrame(cluster_metrics)
    df_clusters["dc"] = df_clusters["datacenter"].apply(lambda x: x["name"])
    
    # Get unique datacenters
    all_dcs = sorted(df_servers["dc"].unique())
    
    # Build overview_list (one entry per datacenter)
    overview_list = []
    for dc in all_dcs:
        # Clusters in this DC
        clusters_count = int((df_clusters["dc"] == dc).sum())
        
        # Servers in this DC
        dc_servers = df_servers[df_servers["dc"] == dc]
        total_servers = int(len(dc_servers))
        spare_servers = int((dc_servers["role"] == "spare").sum())
        spare_pct = (spare_servers / total_servers * 100.0) if total_servers > 0 else 0.0
        
        # VNF types present in this DC
        vnf_count = sum(1 for vnf in vnf_types if any(inst["datacenter"] == dc for inst in vnf["vnf_instancevnf"]))
        
         # VMs in this DC
        vms_count = len([vm for vm in vm_info if vm["vnfc_vm"]["vnfc_instancevnf"]["datacenter"] == dc])
        
        overview_list.append({
            "dc": dc.replace("_", " "),
            "clusters": clusters_count,
            "total_servers": total_servers,
            "spare_servers": spare_servers,
            "spare_pct": f"{spare_pct:.2f}",
            "vnf_types": vnf_count,
            "vms": vms_count,
        })
    
    # Build total_overview (sum of all DCs)
    total_overview = {
        "dc": "Total",
        "clusters": sum(o["clusters"] for o in overview_list),
        "total_servers": sum(o["total_servers"] for o in overview_list),
        "spare_servers": sum(o["spare_servers"] for o in overview_list),
        "vnf_types": len(vnf_types),
        "vms": sum(o["vms"] for o in overview_list),
    }
    total_spare_pct = (total_overview["spare_servers"] / total_overview["total_servers"] * 100.0) if total_overview["total_servers"] > 0 else 0.0
    total_overview["spare_pct"] = f"{total_spare_pct:.2f}"
    
    return overview_list, total_overview