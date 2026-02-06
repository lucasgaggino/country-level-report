from mat import MATClient

import queries


def _to_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def _to_int(x):
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None

def get_region_sites(job, region):
    mat_client = MATClient()
    
    limit = 500
    offset = 0
    data = []
    query_data = ["OK"]
    
    try:
        while query_data:
            variables = {
                "region": region,
                "limit": limit,
                "offset": offset,
            }
            
            query_result = mat_client.graphQL.execute(operation=queries.GET_REGION_SITES, variables=variables)
            query_data = query_result.get("data", {}).get("Resources_Datacenter", [])
            
            offset += limit
            data += query_data
        
        job.log.info(f"Fetched {len(data)} Region Sites.")
        
        sites = [item.get('site') for item in data]
        return sites
        
    except Exception as e:
        job.log.error(f"Failed to query metrics: {e}")
        raise e


def get_datacenter_metrics(job, sites, since):
    mat_client = MATClient()
    
    limit = 500
    offset = 0
    data = []
    query_data = ["OK"]
    
    try:
        while query_data:
            variables = {
                "sites": sites,
                "since": since,
                "limit": limit,
                "offset": offset,
            }
            
            query_result = mat_client.graphQL.execute(operation=queries.GET_DC_METRICS, variables=variables)
            query_data = query_result.get("data", {}).get("Metrics_DC_Metrics_Monthly", [])
            
            offset += limit
            data += query_data
        
        job.log.info(f"Fetched {len(data)} DC Metrics.")
        
        for metric in data:
            name = metric.get('datacenter', {}).get('name','').replace("_"," ")
            cpu_raw = _to_int(metric.get("capacity_cpu_raw"))
            cpu_spare = _to_int(metric.get("capacity_spare_cpu"))
    
            mem_raw = _to_float(metric.get("capacity_mem_raw"))
            mem_spare = _to_float(metric.get("capacity_spare_mem"))
    
            disk_raw = _to_float(metric.get("capacity_disk_raw"))
            disk_spare = _to_float(metric.get("capacity_spare_disk"))
            
            metric['datacenter'] = name
            metric["capacity_cpu_net"] = (cpu_raw - cpu_spare) if (cpu_raw is not None and cpu_spare is not None) else None
            metric["capacity_mem_net"] = (mem_raw - mem_spare) if (mem_raw is not None and mem_spare is not None) else None
            metric["capacity_disk_net"] = (disk_raw - disk_spare) if (disk_raw is not None and disk_spare is not None) else None
        
        return data
    
    except Exception as e:
        job.log.error(f"Failed to query metrics: {e}")
        raise e


def get_region_metrics(job, region, since):
    mat_client = MATClient()
    
    limit = 500
    offset = 0
    data = []
    query_data = ["OK"]
    
    try:
        while query_data:
            variables = {
                "region": region,
                "since": since,
                "limit": limit,
                "offset": offset,
            }
            
            query_result = mat_client.graphQL.execute(operation=queries.GET_REGION_METRICS, variables=variables)
            query_data = query_result.get("data", {}).get("Metrics_Region_Metrics_Monthly", [])
            
            offset += limit
            data += query_data
        
        job.log.info(f"Fetched {len(data)} Region Metrics.")
        
        for metric in data:
            cpu_raw = _to_int(metric.get("capacity_cpu_raw"))
            cpu_spare = _to_int(metric.get("capacity_spare_cpu"))
    
            mem_raw = _to_float(metric.get("capacity_mem_raw"))
            mem_spare = _to_float(metric.get("capacity_spare_mem"))
    
            disk_raw = _to_float(metric.get("capacity_disk_raw"))
            disk_spare = _to_float(metric.get("capacity_spare_disk"))
    
            metric["capacity_cpu_net"] = (cpu_raw - cpu_spare) if (cpu_raw is not None and cpu_spare is not None) else None
            metric["capacity_mem_net"] = (mem_raw - mem_spare) if (mem_raw is not None and mem_spare is not None) else None
            metric["capacity_disk_net"] = (disk_raw - disk_spare) if (disk_raw is not None and disk_spare is not None) else None
        
        return data
    
    except Exception as e:
        job.log.error(f"Failed to query metrics: {e}")
        raise e


def get_cluster_metrics(job, sites, since):
    mat_client = MATClient()
    
    limit = 500
    offset = 0
    data = []
    query_data = ["OK"]
    
    try:
        while query_data:
            variables = {
                "sites": sites,
                "since": since,
                "limit": limit,
                "offset": offset,
            }
            
            query_result = mat_client.graphQL.execute(operation=queries.GET_CLUSTER_METRICS,variables=variables)
            query_data = query_result.get("data", {}).get("Metrics_Cluster_Metrics_Monthly", [])
            
            offset += limit
            data += query_data
            
        cluster_names = list(set([item['cluster'] for item in data]))
        job.log.info(f"Fetched {len(data)} Cluster Metrics.")
        
        roles_data = []
        query_data = ["OK"]
        
        while query_data:
            variables = {
                "names": cluster_names,
                "sites": sites,
                "limit": limit,
                "offset": offset,
            }
            
            query_result = mat_client.graphQL.execute(operation=queries.GET_ROLES, variables=variables)
            query_data = query_result.get('data',{}).get('Resources_Cluster',[])
            
            offset += limit
            roles_data += query_data
        
        # Cruzar los datos (Join en memoria)
        # Convertir roles a un diccionario para búsqueda rápida: {'cluster1': 'active', 'cluster2': 'internal'}
        roles_map = {role['clusterId']: role['role'] for role in roles_data }
        
        # Agregar el rol a cada métrica
        for item in data:
            item['role'] = roles_map.get(item['cluster'], 'Unknown')
            
            if "internal" in item["cluster"]:
                item["cluster"] = "internal"
        
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query metrics: {e}")
        raise e


def get_server_info(job, sites):
    mat_client = MATClient()
    
    limit = 500
    offset = 0
    data = []
    query_data = ["OK"]
    
    try:
        while query_data:
            variables = {
                "sites": sites,
                "limit": limit,
                "offset": offset,
            }
            
            query_result = mat_client.graphQL.execute(operation=queries.GET_SERVER, variables=variables)
            query_data = query_result.get("data", {}).get("Resources_Server", [])
            
            offset += limit
            data += query_data
        
        job.log.info(f"Fetched {len(data)} Servers")
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query resources: {e}")
        raise e


def get_vnf_types(job, sites):
    mat_client = MATClient()
    
    limit = 500
    offset = 0
    data = []
    query_data = ["OK"]
    
    try:
        while query_data:
            variables = {
                "sites": sites,
                "limit": limit,
                "offset": offset,
            }
            
            query_result = mat_client.graphQL.execute(operation=queries.GET_VNF_TYPES,variables=variables)
            query_data = query_result.get("data", {}).get("Resources_VNF", [])
        
            offset += limit
            data += query_data
            
        job.log.info(f"Fetched {len(data)} VNF Types")
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query resources: {e}")
        raise e


def get_vnf_metrics(job, sites, since):
    mat_client = MATClient()
    
    limit = 500
    offset = 0
    data = []
    query_data = ["OK"]
    
    try:
        while query_data:
            variables = {
                "sites": sites,
                "since": since,
                "limit": limit,
                "offset": offset,
            }
            
            query_result = mat_client.graphQL.execute(operation=queries.GET_VNF_METRICS, variables=variables)
            query_data = query_result.get("data", {}).get("Metrics_VNF_Metrics_Monthly", [])
            
            offset += limit
            data += query_data
        
        job.log.info(f"Fetched {len(data)} VNF Metrics.")
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query metrics: {e}")
        raise e


def get_vms_info(job, sites):
    mat_client = MATClient()
    
    limit = 500
    offset = 0
    data = []
    query_data = ["OK"]
    
    try:
        while query_data:
            variables = {
                "sites": sites,
                "limit": limit,
                "offset": offset,
            }
            
            query_result = mat_client.graphQL.execute(operation=queries.GET_VMS,variables=variables)
            query_data = query_result.get("data", {}).get("Resources_VM", [])
            
            offset += limit
            data += query_data
        
        job.log.info(f"Fetched {len(data)} VMs")
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query resources: {e}")
        raise e


def get_vnfc_metrics(job, sites, since):
    mat_client = MATClient()
    
    limit = 500
    offset = 0
    data = []
    query_data = ["OK"]
    
    try:
        while query_data:
            variables = {
                "sites": sites,
                "since": since,
                "limit": limit,
                "offset": offset,
            }
            
            query_result = mat_client.graphQL.execute(operation=queries.GET_VNFC,variables=variables)
            query_data = query_result.get("data", {}).get("Metrics_VNFC_Metrics_Monthly", [])
            
            offset += limit
            data += query_data
            
        job.log.info(f"Fetched {len(data)} VNFC Metrics.")
        
        data = sorted(
            data,
            key=lambda r: (
                (((r.get("vnfc") or {}).get("vnfc_instancevnf") or {}).get("vnf_instancevnf") or {}).get("vnf") or "",
                ((r.get("vnfc") or {}).get("role") or "")
            )
        )
        
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query metrics: {e}")
        raise e


def get_nas_metrics(job, sites):
    mat_client = MATClient()
    
    limit = 500
    offset = 0
    data = []
    query_data = ["OK"]
    
    try:
        while query_data:
            variables = {
                "sites": sites,
                "limit": limit,
                "offset": offset,
            }
            
            query_result = mat_client.graphQL.execute(operation=queries.GET_NAS,variables=variables)
            query_data = query_result.get("data", {}).get("Resources_Storage_Pool", [])
            
            offset += limit
            data += query_data
            
        job.log.info(f"Fetched {len(data)} NAS Metrics.")
        
        # Calculo porcentaje
        total_capacity_used_tib = 0
        total_capacity_total_tib = 0
        for item in data:
            item['datacenter'] = item.get('pools',{}).get('node_a',{}).get('rack_server',{}).get('dc','').replace("_", " ")
            if item.get("capacity_total_tib"):
                total_capacity_used_tib += item["capacity_used_tib"]
                total_capacity_total_tib += item["capacity_total_tib"]
                item["capacity_used_pct"] = (item["capacity_used_tib"] / item["capacity_total_tib"]) * 100
            else:
                item["capacity_used_pct"] = None
        
        total_data = {
            "capacity_used_tib": total_capacity_used_tib,
            "capacity_total_tib": total_capacity_total_tib,
            "capacity_used_pct": (total_capacity_used_tib/total_capacity_total_tib)*100,
        }
        
        return data, total_data
        
    except Exception as e:
        job.log.error(f"Failed to query metrics: {e}")
        raise e


def get_vm_disk_metrics(job, sites, since):
    mat_client = MATClient()
    
    limit = 500
    offset = 0
    data = []
    query_data = ["OK"]
    
    try:
        while query_data:
            variables = {
                "sites": sites,
                "since": since,
                "limit": limit,
                "offset": offset,
            }
            
            query_result = mat_client.graphQL.execute(operation=queries.GET_VM_DISK_METRICS,variables=variables)
            query_data = query_result.get("data", {}).get("Metrics_VM_disk_metrics", [])
            
            offset += limit
            data += query_data
        
        job.log.info(f"Fetched {len(data)} VM Disk Metrics.")
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query metrics: {e}")
        raise e


def get_server_disk_metrics(job, sites, since):
    mat_client = MATClient()
    
    limit = 500
    offset = 0
    data = []
    query_data = ["OK"]
    
    try:
        while query_data:
            variables = {
                "sites": sites,
                "since": since,
                "limit": limit,
                "offset": offset,
            }
            
            query_result = mat_client.graphQL.execute(operation=queries.GET_SERVER_DISK_METRICS,variables=variables)
            query_data = query_result.get("data", {}).get("Metrics_Server_disk_Metrics", [])
            
            offset += limit
            data += query_data
        
        job.log.info(f"Fetched {len(data)} Server Disk Metrics.")
        
        for item in data:
            item["server"]["cluster_server"]["name"] = item["server"]["cluster_server"]["clusterId"]
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query metrics: {e}")
        raise e


def get_cluster_spare_active_metrics(job, sites, since):
    mat_client = MATClient()
    
    limit = 500
    offset = 0
    data = []
    query_data = ["OK"]
    
    try:
        while query_data:
            variables = {
                "sites": sites,
                "since": since,
                "limit": limit,
                "offset": offset,
            }
            
            query_result = mat_client.graphQL.execute(operation=queries.GET_CLUSTER_SPARE_ACTIVE_METRICS,variables=variables)
            query_data = query_result.get("data", {}).get("Metrics_Cluster_Metrics_Spare_Active", [])
            
            offset += limit
            data += query_data
        
        job.log.info(f"Fetched {len(data)} Cluster spare/active Metrics.")
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query metrics: {e}")
        raise e


def get_redundancy_info(job, region):
    mat_client = MATClient()
    
    limit = 500
    offset = 0
    data = []
    query_data = ["OK"]
    
    try:
        while query_data:
            variables = {
                "region": f"%{region}%",
                "limit": limit,
                "offset": offset,
            }
            
            query_result = mat_client.graphQL.execute(operation=queries.GET_REDUNDANCY,variables=variables)
            query_data = query_result.get("data", {}).get("Resources_VNFc_Redundancy", [])
            
            offset += limit
            data += query_data
        
        job.log.info(f"Fetched {len(data)} VNFc Redundancy Groups")
        
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query resources: {e}")
        raise e