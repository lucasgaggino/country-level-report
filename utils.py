
def _run_pdflatex_and_log(job, cmd, work_dir, expected_log_path, label, timeout_s=300):
    
    import subprocess
    import os
    import time
    
    job.log.info(f"{label}: running pdflatex (timeout={timeout_s}s) cmd={cmd}")
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            cwd=work_dir,
            check=False,
            timeout=timeout_s,
        )
        dt = time.time() - t0
        job.log.info(f"{label}: pdflatex finished rc={p.returncode} elapsed={dt:.1f}s")

        out = (p.stdout or b"").decode("utf-8", errors="replace")
        err = (p.stderr or b"").decode("utf-8", errors="replace")

        # Logueo un resumen y el final (para no inundar)
        if out.strip():
            job.log.warning(f"{label}: stdout (tail 2000 chars):\n{out[-2000:]}")
        if err.strip():
            job.log.warning(f"{label}: stderr (tail 2000 chars):\n{err[-2000:]}")

        # Si existe el .log de LaTeX, logueo el final también (lo más útil)
        if os.path.exists(expected_log_path):
            try:
                with open(expected_log_path, "r", encoding="utf-8", errors="replace") as f:
                    latex_log = f.read()
                job.log.warning(f"{label}: latex .log (tail 4000 chars):\n{latex_log[-4000:]}")
            except Exception as e:
                job.log.warning(f"{label}: could not read latex log: {e}")

        return p

    except subprocess.TimeoutExpired as e:
        dt = time.time() - t0
        job.log.error(f"{label}: pdflatex TIMEOUT after {dt:.1f}s")

        # Si había output parcial, lo logueo
        try:
            out = (e.stdout or b"").decode("utf-8", errors="replace")
            err = (e.stderr or b"").decode("utf-8", errors="replace")
            if out.strip():
                job.log.error(f"{label}: stdout (tail 2000 chars):\n{out[-2000:]}")
            if err.strip():
                job.log.error(f"{label}: stderr (tail 2000 chars):\n{err[-2000:]}")
        except Exception:
            pass

        # Intento loguear el final del .log si existe
        if os.path.exists(expected_log_path):
            try:
                with open(expected_log_path, "r", encoding="utf-8", errors="replace") as f:
                    latex_log = f.read()
                job.log.error(f"{label}: latex .log (tail 4000 chars):\n{latex_log[-4000:]}")
            except Exception as ex:
                job.log.error(f"{label}: could not read latex log after timeout: {ex}")

        raise


def get_dc_metrics(job,mat_client,site,time):
    
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
    
    dc_query = """
    query GetDCMetricsDistinct($dcName: String!,  $targetTime: timestamptz!) {
      Metrics_DC_Metrics_Monthly(
        distinct_on: [datacenter_id],
        order_by: [{datacenter_id: asc}],
        where: {
          datacenter: {
            name: { _eq: $dcName } 
          },
          time: { _eq: $targetTime }
        }
      ) {
        capacity_cpu_raw
        capacity_disk_raw
        capacity_mem_raw
        capacity_spare_cpu
        capacity_spare_cpu_pct
        capacity_spare_disk
        capacity_spare_disk_pct
        capacity_spare_mem
        capacity_spare_mem_pct
        datacenter_id
        efect_capacity_cpu
        efect_capacity_cpu_pct
        efect_capacity_disk
        efect_capacity_disk_pct
        efect_capacity_mem
        efect_capacity_mem_pct
        efect_resv_cpu_pct
        efect_resv_disk
        efect_resv_disk_pct
        efect_resv_mem_pct
        efect_util_cpu
        efect_util_disk
        efect_util_disk_pct
        efect_util_failover_cpu
        efect_util_failover_disk
        efect_util_failover_mem
        efect_util_mem
        region
        time
      }
    }
    """
    variables = {"dcName": site,"targetTime":time}
    
    try:
        query_result = mat_client.graphQL.execute(operation=dc_query,variables=variables)

        data = query_result.get("data", {}).get("Metrics_DC_Metrics_Monthly", [])
        job.log.info(f"Fetched {len(data)} DC Metrics.")
        
        for d in data:
            cpu_raw = _to_int(d.get("capacity_cpu_raw"))
            cpu_spare = _to_int(d.get("capacity_spare_cpu"))

            mem_raw = _to_float(d.get("capacity_mem_raw"))
            mem_spare = _to_float(d.get("capacity_spare_mem"))

            disk_raw = _to_float(d.get("capacity_disk_raw"))
            disk_spare = _to_float(d.get("capacity_spare_disk"))

            d["capacity_cpu_net"] = (cpu_raw - cpu_spare) if (cpu_raw is not None and cpu_spare is not None) else None
            d["capacity_mem_net"] = (mem_raw - mem_spare) if (mem_raw is not None and mem_spare is not None) else None
            d["capacity_disk_net"] = (disk_raw - disk_spare) if (disk_raw is not None and disk_spare is not None) else None
        
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query metrics: {e}")
        raise e

def get_cluster_metrics(job,mat_client,site,time):
    
    cluster_query = """
    query GetClusterMetricsDistinct($dcName: String!, $targetTime: timestamptz!) {
      Metrics_Cluster_Metrics_Monthly(
        distinct_on: [cluster],
        order_by: [{cluster: asc}],
        where: {
          time: { _eq: $targetTime },
          datacenter: {
            name: { _eq: $dcName }
          }
        }
      ) {
        time
        cluster
        datacenter {
          site
          name
          admin
          mat_pk
          country
          dc
          region
          vendor
        }
        capacity_cpu_raw
        efect_capacity_cpu
        efect_capacity_cpu_pct
        efect_resv_cpu_pct
        efect_util_cpu
        capacity_mem_raw
        efect_capacity_mem
        efect_capacity_mem_pct
        efect_resv_mem_pct
        efect_util_mem
        capacity_disk_raw
        efect_resv_disk
        efect_util_disk
        efect_resv_cpu
        efect_resv_mem
        efect_util_cpu_pct
        efect_util_disk_pct
        efect_util_mem_pct
        mem_p95_pct
        cpu_p95_pct
      }
    }
    """
    
    variables = {"dcName": site,"targetTime":time}
    try:
        query_result = mat_client.graphQL.execute(operation=cluster_query,variables=variables)

        data = query_result.get("data", {}).get("Metrics_Cluster_Metrics_Monthly", [])
        cluster_names = list(set([d['cluster'] for d in data]))
        job.log.info(f"Fetched {len(data)} Cluster Metrics.")
        
        query_roles = """
        query GetRolesByClusterAndSite($names: [String!], $dcName: String!) {
          Resources_Cluster(
            where: {
              clusterId: { _in: $names },           # Filtra los clústers que están en la lista obtenida previamente
              datacenter_cluster: {            # Navega la relación hacia el Datacenter
                name: { _eq: $dcName }         # Filtra donde el nombre del DC sea igual a la variable
              }
            }
          ) {
            clusterId
            role
          }
        }
        """
        roles_data = mat_client.graphQL.execute(operation=query_roles, variables={"names": cluster_names, "dcName": site})
        
        # 4. Cruzar los datos (Join en memoria)
        # Convertir roles a un diccionario para búsqueda rápida: {'cluster1': 'active', 'cluster2': 'internal'}
        roles_map = { r['clusterId']: r['role'] for r in roles_data['data']['Resources_Cluster'] }
        
        # Agregar el rol a cada métrica
        for d in data:
            d['role'] = roles_map.get(d['cluster'], 'Unknown')
            
            if "internal" in d["cluster"]:
                d["cluster"] = "internal"
        
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query metrics: {e}")
        raise e

def get_vnf_metrics(job,mat_client,site,time):
    
    vnf_query = """
    query GetVNFMetricsDistinct($dcName: String!, $targetTime: timestamptz!) {
      Metrics_VNF_Metrics_Monthly(
        distinct_on: [vnf_id],
        order_by: [{vnf_id: asc}, {time: desc}],
        where: {
          vnf: {
            datacenter: {_eq: $dcName}
          }, 
          time: {_eq: $targetTime}
        }
      ) {
        vnf_id
        cpu_p95
        efect_util_cpu
        efect_util_cpu_pct
        efect_util_disk
        efect_util_disk_pct
        efect_util_mem
        efect_util_mem_pct
        idle_cpu
        idle_cpu_pct
        idle_disk
        idle_disk_pct
        idle_mem
        idle_mem_pct
        ram_p95
        resv_disk
        resv_cpu
        resv_mem
        mem_p95_raw
        cpu_p95_raw
        vnf {
          datacenter
          name
        }
        time
      }
    }
    """

    variables = {"dcName": site,"targetTime":time}
    
    try:
        query_result = mat_client.graphQL.execute(operation=vnf_query,variables=variables)

        data = query_result.get("data", {}).get("Metrics_VNF_Metrics_Monthly", [])
        job.log.info(f"Fetched {len(data)} VNF Metrics.")
        
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query metrics: {e}")
        raise e

def get_sites(job,mat_client,region):
    
    sites_query = """
    query get_sites($region:String){
  Resources_Datacenter(where:{_and:[
    {region:{_eq:$region}}
    {site:{_nin:["Morales","Portales"]}}
  ]}) {
    site
    region
  }
}
    """

    variables = {"region": region}
    
    try:
        query_result = mat_client.graphQL.execute(operation=sites_query,variables=variables)

        data = query_result.get("data", {}).get("Resources_Datacenter", [])
        job.log.info(f"Fetched {len(data)} VNF Metrics.")
        
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query metrics: {e}")
        raise e


def top5_cpu_p95(data):
    
    import pandas as pd
    
    df = pd.DataFrame(data)
    
    # Aplanar datacenter desde la columna vnf
    df["name"] = df["vnf"].apply(
        lambda x: x.get("name").split("_")[0]
        if isinstance(x, dict) and x.get("name")
        else None
    )


    # Filtrar cpu_p95 válidos
    df = df[df["cpu_p95"].notna()]

    # Ordenar descendente y tomar top 5
    top5 = (
        df.sort_values("cpu_p95", ascending=False)
          .head(5)
          .reset_index(drop=True)
    )

    return top5[[
        "vnf_id",
        "name",
        "cpu_p95",
        "ram_p95",
        "efect_util_cpu_pct",
        "efect_util_mem_pct",
        "efect_util_disk_pct"
    ]]
    
def get_server_info(job,mat_client,site):
    
    server_query = """
    query GetServerMetrics($dcName: String!) {
      Resources_Server(
        where: {
          rack_server: {
            dc_rack: {
              name: { _eq: $dcName }
            }
          }
        }
      ) {
        hostname
        role
        cluster_server{
          name
          clusterId
        }
        storage_total_gb
      }
    }
    """
    variables = {"dcName": site}
    
    try:
        query_result = mat_client.graphQL.execute(operation=server_query,variables=variables)

        data = query_result.get("data", {}).get("Resources_Server", [])
        job.log.info(f"Fetched {len(data)} Servers")
        
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query resources: {e}")
        raise e
        
def get_vnf_types(job,mat_client,site):
    
    vnf_types_query = """
    query GetVNFTypes($dcName: String!) {
      Resources_VNF(where: {vnf_instancevnf: {datacenter: {_eq: $dcName}}}) {
        vnf
      }
    }
    """
    
    variables = {"dcName": site}
    
    try:
        query_result = mat_client.graphQL.execute(operation=vnf_types_query,variables=variables)

        data = query_result.get("data", {}).get("Resources_VNF", [])
        job.log.info(f"Fetched {len(data)} VNF Types")
        
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query resources: {e}")
        raise e
        
def get_vms_info(job,mat_client,site):
    
    vm_query = """
    query GetVMInfo($dcName: String!) {
      Resources_VM(where: {vnfc_vm: {vnfc_instancevnf: {datacenter: {_eq: $dcName}}}}) {
        name
        vm
        hypervisor_vm {
          cluster
        }
        vnfc_vm {
          vnfc_instancevnf {
            vnf_instancevnf {
              name
            }
          }
        }
      }
    }
    """
    
    variables = {"dcName": site}
    
    try:
        query_result = mat_client.graphQL.execute(operation=vm_query,variables=variables)

        data = query_result.get("data", {}).get("Resources_VM", [])
        job.log.info(f"Fetched {len(data)} VMs")
        
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query resources: {e}")
        raise e
        
def get_vnfc_metrics(job,mat_client,site,time):
    
    vnfc_query = """
    query GetVNFCMetricsByInstanceVNF($dcName: String!, $targetTime: timestamptz!) {
      Metrics_VNFC_Metrics_Monthly(
        distinct_on: [vnfc_id],
        order_by: [{vnfc_id: asc}], 
        where: {
          time: {_eq: $targetTime}, 
          vnfc: {
            vnfc_instancevnf: {
              datacenter: {_eq: $dcName}
            }
          }
        }
      ) {
        efect_util_disk_pct
        efect_util_cpu_pct
        efect_util_mem_pct
        mem_p95_pct
        cpu_p95_pct
        vnfc {
          vnfc
          role
          vnfc_vm {
            name
          }
          vnfc_instancevnf {
            name
            vnf_instancevnf {
              vnf
            }
          }
        }
      }
    }
    """
    
    variables = {"dcName": site,"targetTime":time}
    
    try:
        query_result = mat_client.graphQL.execute(operation=vnfc_query,variables=variables)

        data = query_result.get("data", {}).get("Metrics_VNFC_Metrics_Monthly", [])
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

def get_redundancy_info(job,mat_client,site,region):
    
    redundancy_query =  """
    query GetRedundancyInfo($region: String!) {
      Resources_VNFc_Redundancy(where: {groupName: {_ilike: $region}}) {
        groupName
        red_vnfc {
          vnfc
          vnfc_instancevnf {
            datacenter
            vnf_instancevnf {
              name
            }
          }
        }
        redundancyType
      }
    }
    """
    
    variables = {"region": f"%{region}%"}
    
    try:
        query_result = mat_client.graphQL.execute(operation=redundancy_query,variables=variables)

        data = query_result.get("data", {}).get("Resources_VNFc_Redundancy", [])
        job.log.info(f"Fetched {len(data)} VNFc Redundancy Groups")
        
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query resources: {e}")
        raise e
        
def build_geo_redundancy_rows(job, raw_groups, dc_names):
    """
    raw_groups: lista de dicts con keys: groupName, red_vnfc, redundancyType
    dc_name: el DC actual (ej: "Mayo")

    Devuelve: list[dict] con:
      - vnf (str)
      - sites (str)        (ya formateado "A, B, C")
      - redundancy (str)   (ya mapeado a label)
    """
    # imports locales (MAT-friendly)
    import re

    # --- Redundancy mapping pedido ---
    redundancy_map = {
        "pool": "Pool",
        "activeActive": "Activo-Activo",
        "activeStandBy": "Activo-Pasivo",
    }

    def map_redundancy(value):
        if not value:
            return "n/a"
        v = str(value).strip()
        return redundancy_map.get(v, v)

    # --- Normalización de sitios ---
    def normalize_site(site_raw):
        if not site_raw:
            return None

        s = str(site_raw).strip()

        # Excepciones pedidas
        if s == "Revolucion_Nacional":
            s = "Revolucion"
        elif s in ("Nextengo_Nacional", "Nextengo_Regional"):
            s = "Nextengo"

        # Presentación: underscores -> espacios
        s = s.replace("_", " ").strip()

        # Respetar el dc_name tal cual si matchea (case-insensitive)
        for dc_name in dc_names:
            if s.lower() == str(dc_name).strip().lower():
                return dc_name

        return s

    def title_case_spanish(s):
        # "Revolucion" -> "Revolución" (acento)
        if s == "Revolucion":
            return "Revolución"
        return s

    # --- Extractores ---
    def extract_vnf(item):
        # Viene de: item["red_vnfc"][0]["vnfc_instancevnf"]["vnf_instancevnf"]["name"]
        try:
            red = item.get("red_vnfc") or []
            if not red:
                return "N/A"
            v = ((red[0].get("vnfc_instancevnf") or {}).get("vnf_instancevnf") or {})
            name = v.get("name")
            return str(name) if name else "N/A"
        except Exception:
            return "N/A"

    def extract_sites(item):
        sites = []

        group_name = item.get("groupName") or ""

        # Regla: si groupName contiene Revolucion -> agregar Revolución
        # (incluye Revolucion_Nacional porque contiene "Revolucion")
        if "Revolucion" in group_name:
            sites.append("Revolución")

        # NUEVO: si aparece Mayo o Madero en groupName, agregarlos aunque no estén en hijos
        if "Mayo" in group_name:
            sites.append(normalize_site("Mayo"))
        if "Madero" in group_name:
            sites.append(normalize_site("Madero"))

        # Excepción: Nextengo_Nacional / Nextengo_Regional => Nextengo
        if "Nextengo_Nacional" in group_name or "Nextengo_Regional" in group_name:
            sites.append(normalize_site("Nextengo_Nacional"))  # normalize_site -> "Nextengo"

        # Regla original: sumar todos los datacenter de red_vnfc
        red = item.get("red_vnfc") or []
        for r in red:
            dc_raw = ((r.get("vnfc_instancevnf") or {}).get("datacenter"))
            dc_norm = normalize_site(dc_raw)
            if dc_norm:
                sites.append(dc_norm)

        # Normalizar acentos / nombres
        sites = [title_case_spanish(s) for s in sites if s]

        # Unicos preservando orden
        seen = set()
        uniq = []
        for s in sites:
            if s not in seen:
                uniq.append(s)
                seen.add(s)

        return ", ".join(uniq) if uniq else "n/a"

    # --- Build rows (por item) ---
    rows = []
    for item in raw_groups or []:
        vnf = extract_vnf(item)
        redundancy = map_redundancy(item.get("redundancyType"))
        sites = extract_sites(item)

        rows.append(
            {
                "vnf": vnf,
                "sites": sites,
                "redundancy": redundancy,
            }
        )

    # --- Consolidar por (vnf, redundancy) para unir sitios de múltiples filas ---
    grouped = {}
    for r in rows:
        key = (r["vnf"], r["redundancy"])
        if key not in grouped:
            grouped[key] = {"vnf": r["vnf"], "redundancy": r["redundancy"], "sites_list": []}
        grouped[key]["sites_list"].append(r["sites"])

    consolidated = []
    for (_k, g) in grouped.items():
        all_sites = []
        for sites_str in g["sites_list"]:
            for s in [x.strip() for x in str(sites_str).split(",")]:
                if s and s.lower() != "n/a":
                    all_sites.append(s)

        # únicos preservando orden
        seen = set()
        uniq = []
        for s in all_sites:
            if s not in seen:
                uniq.append(s)
                seen.add(s)

        sites_final = ", ".join(uniq) if uniq else "n/a"

        consolidated.append(
            {
                "vnf": g["vnf"],
                "sites": sites_final,
                "redundancy": g["redundancy"],
            }
        )

    # Orden opcional (alfabético por VNF)
    consolidated.sort(key=lambda x: (x.get("vnf") or ""))
    return consolidated

def get_nas_metrics(job,mat_client,site):
    
    nas_query = """
    query GetNASMetrics($dcName: String!) {
      Resources_Storage_Pool(
        where: {
          pools: {
            node_a: {
              rack_server: {
                dc: { _eq: $dcName }
              }
            }
          }
        }
      ) {
        capacity_total_tib
        capacity_used_tib
      }
    }
    """
    variables = {"dcName": site}
    
    try:
        query_result = mat_client.graphQL.execute(operation=nas_query,variables=variables)

        data = query_result.get("data", {}).get("Resources_Storage_Pool", [])
        job.log.info(f"Fetched {len(data)} NAS Metrics.")
        
        # Calculo porcentaje
        for d in data:
            if d.get("capacity_total_tib"):
                d["capacity_used_pct"] = (d["capacity_used_tib"] / d["capacity_total_tib"]) * 100
            else:
                d["capacity_used_pct"] = None
        
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query metrics: {e}")
        raise e

def build_top10_idle_tables(job, vnf_metrics, dc_name):
    """
    vnf_metrics: list[dict] como el que pegaste (cada item tiene vnf.name, resv_*, idle_*, efect_util_*_pct)
    dc_name: "Mayo" (solo para filtrar si querés)

    Devuelve un dict con 3 listas (list[dict]) listas para el template:
      - top10_idle_cpu
      - top10_idle_mem
      - top10_idle_disk
    """
    import pandas as pd

    def _vnf_type_from_name(v):
        # "PCRF01_Mayo" -> "PCRF01"
        if isinstance(v, str) and v:
            return v.split("_")[0]
        return "N/A"

    def _num(x):
        # Convierte a float si se puede; si no, NaN
        try:
            return float(x)
        except Exception:
            return None

    def _fmt(x, decimals=1, na="N/A"):
        if x is None:
            return na
        try:
            return f"{float(x):.{decimals}f}"
        except Exception:
            return na

    def _fmt_pct(x, decimals=2, na="N/A"):
        if x is None:
            return na
        try:
            return f"{float(x):.{decimals}f}"
        except Exception:
            return na

    df = pd.DataFrame(vnf_metrics or [])

    if df.empty:
        return {"top10_idle_cpu": [], "top10_idle_mem": [], "top10_idle_disk": []}

    # Aplanar vnf.name y sacar tipo
    df["vnf_name"] = df["vnf"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)
    df["vnf_type"] = df["vnf_name"].apply(_vnf_type_from_name)

    # Normalizar columnas numéricas relevantes
    for c in [
        "resv_cpu", "idle_cpu", "idle_cpu_pct", "efect_util_cpu_pct",
        "resv_mem", "idle_mem", "idle_mem_pct", "efect_util_mem_pct",
        "resv_disk", "idle_disk", "idle_disk_pct", "efect_util_disk_pct",
        "mem_p95_raw", "cpu_p95_raw", "ram_p95", "cpu_p95"
    ]:
        if c in df.columns:
            df[c] = df[c].apply(_num)

    # --- CPU: calcular util efectiva "absoluta" si hace falta ---
    # Regla: util_cpu_abs = resv_cpu - idle_cpu  (si ambos están)
    df["util_cpu_abs"] = df.apply(
        lambda r: (r["resv_cpu"] - r["idle_cpu"])
        if (r.get("resv_cpu") is not None and r.get("idle_cpu") is not None)
        else None,
        axis=1
    )

    # --- MEM: util efectiva absoluta ---
    df["util_mem_abs"] = df.apply(
        lambda r: (r["resv_mem"] - r["idle_mem"])
        if (r.get("resv_mem") is not None and r.get("idle_mem") is not None)
        else None,
        axis=1
    )

    # --- DISK: util efectiva absoluta ---
    df["util_disk_abs"] = df.apply(
        lambda r: (r["resv_disk"] - r["idle_disk"])
        if (r.get("resv_disk") is not None and r.get("idle_disk") is not None)
        else None,
        axis=1
    )

    # Helpers para armar columnas tipo "985.7 (80.01%)"
    def _fmt_value_pct(value, pct, value_decimals=1, pct_decimals=2):
        if value is None and pct is None:
            return "N/A"
        if value is None:
            return f"N/A ({_fmt_pct(pct, pct_decimals)}\\%)"
        if pct is None:
            return f"{_fmt(value, value_decimals)} (N/A\\%)"
        return f"{_fmt(value, value_decimals)} ({_fmt_pct(pct, pct_decimals)}\\%)"

    # ======================
    # TOP10 CPU OCIOSA
    # ======================
    cpu_df = df.copy()
    # Orden: mayor idle_cpu (si None, al final)
    cpu_df = cpu_df.sort_values(by=["idle_cpu"], ascending=False, na_position="last").head(10)

    top10_idle_cpu = []
    for _, r in cpu_df.iterrows():
        top10_idle_cpu.append({
            "vnf_type": r.get("vnf_type") or "N/A",
            "resv_cpu": _fmt(r.get("resv_cpu"), decimals=0),
            "idle_cpu_str": _fmt_value_pct(r.get("idle_cpu"), r.get("idle_cpu_pct"), value_decimals=1, pct_decimals=2),
            "util_cpu_str": _fmt_value_pct(r.get("util_cpu_abs"), r.get("efect_util_cpu_pct"), value_decimals=1, pct_decimals=2),
            "util_local_cpu_str": _fmt_value_pct(
                r.get("cpu_p95_raw"),
                r.get("cpu_p95"),
                value_decimals=1,
                pct_decimals=2
            ),
            "vnf_instance": r.get("vnf_type") or "N/A",
            "idle_cpu": r.get("idle_cpu"),
            "flavor:Vcpus": r.get("resv_cpu"),
            "cpu_p95_raw": r.get("cpu_p95_raw"),
            "cpu_p95": r.get("cpu_p95")
        })


    # ======================
    # TOP10 MEM OCIOSA
    # ======================
    mem_df = df.copy()
    mem_df = mem_df.sort_values(by=["idle_mem"], ascending=False, na_position="last").head(10)

    top10_idle_mem = []
    for _, r in mem_df.iterrows():
        top10_idle_mem.append({
            "vnf_type": r.get("vnf_type") or "N/A",
            "resv_mem": _fmt(
                r.get("resv_mem") / (1024 ** 3) if r.get("resv_mem") is not None else None,
                decimals=1
            ),
            "idle_mem_str": _fmt_value_pct(
                r.get("idle_mem") / (1024 ** 3) if r.get("idle_mem") is not None else None,
                r.get("idle_mem_pct"),
                value_decimals=1,
                pct_decimals=2
            ),
            "util_mem_str": _fmt_value_pct(
                r.get("util_mem_abs") / (1024 ** 3) if r.get("util_mem_abs") is not None else None,
                r.get("efect_util_mem_pct"),
                value_decimals=1,
                pct_decimals=2
            ),
            "util_local_mem_str": _fmt_value_pct(
                r.get("mem_p95_raw") / (1024 ** 3) if r.get("mem_p95_raw") is not None else None,
                r.get("ram_p95"),
                value_decimals=1,
                pct_decimals=2
            ),
            "vnf_instance": r.get("vnf_type") or "N/A",
            "idle_mem": r.get("idle_mem") / (1024 ** 3) if r.get("idle_mem") is not None else None,
            "flavor:Ram": r.get("resv_mem") / (1024 ** 3) if r.get("resv_mem") is not None else None,
            "mem_p95_raw": r.get("mem_p95_raw"),
            "ram_p95": r.get("ram_p95"),
        })


    # ======================
    # TOP10 DISK OCIOSO
    # ======================
    disk_df = df.copy()
    disk_df = disk_df.sort_values(by=["idle_disk"], ascending=False, na_position="last").head(10)

    top10_idle_disk = []
    for _, r in disk_df.iterrows():
        top10_idle_disk.append({
            "vnf_type": r.get("vnf_type") or "N/A",
            "resv_disk": _fmt(
                r.get("resv_disk") / (1024) if r.get("resv_disk") is not None else None,
                decimals=1
            ),
            "idle_disk_str": _fmt_value_pct(
                r.get("idle_disk") / (1024) if r.get("idle_disk") is not None else None,
                r.get("idle_disk_pct"),
                value_decimals=1,
                pct_decimals=2
            ),
            "util_disk_str": _fmt_value_pct(
                r.get("util_disk_abs") / (1024) if r.get("util_disk_abs") is not None else None,
                r.get("efect_util_disk_pct"),
                value_decimals=1,
                pct_decimals=2
            ),
            "vnf_instance": r.get("vnf_type") or "N/A",
            "idle_disk": r.get("idle_disk") / (1024) if r.get("idle_disk") is not None else None,
            "flavor:Disk": r.get("resv_disk") / (1024) if r.get("resv_disk") is not None else None,
        })


    return {
        "top10_idle_cpu": top10_idle_cpu,
        "top10_idle_mem": top10_idle_mem,
        "top10_idle_disk": top10_idle_disk,
    }
    
def get_vm_disk_metrics(job,mat_client,site,time):
    
    vm_disk_query = """
   query GetLatestMetricsPerDisk($dcName: String!) {
      Metrics_VM_disk_metrics(
        distinct_on: [vm_id, disk]
        order_by: [{vm_id: asc}, {disk: asc}]
        where: {vm: {vnfc_vm: {vnfc_instancevnf: {datacenter: {_eq: $dcName}}}}}
      ) {
        time
        disk
        used
        total
        vm {
          name
          hypervisor_vm {
            name
          }
        }
      }
    }
    """
    variables = {"dcName": site}
    
    try:
        query_result = mat_client.graphQL.execute(operation=vm_disk_query,variables=variables)

        data = query_result.get("data", {}).get("Metrics_VM_disk_metrics", [])
        job.log.info(f"Fetched {len(data)} VM Disk Metrics.")
        
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query metrics: {e}")
        raise e
    
def get_server_disk_metrics(job,mat_client,site,time):
    
    server_disk_query = """
    query GetLatestMetricsPerDisk($dcName: String!) {
      Metrics_Server_disk_Metrics(
        distinct_on: [server_id]
        order_by: [{server_id: asc}]
        where: {server: {rack_server: {dc: {_eq: $dcName}}}}
      ) {
        server {
          hostname
          server
          mat_pk
          storage_total_gb
          cluster_server {
            name
          }
        }
        storage_free
        storage_reserved
        storage_system
        storage_total
        storage_used
        time
      }
    }
    """
    variables = {"dcName": site}
    
    try:
        query_result = mat_client.graphQL.execute(operation=server_disk_query,variables=variables)

        data = query_result.get("data", {}).get("Metrics_Server_disk_Metrics", [])
        job.log.info(f"Fetched {len(data)} Server Disk Metrics.")
        
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query metrics: {e}")
        raise e
        
def get_cluster_spare_active_metrics(job,mat_client,site,time):
    
    cluster_spare_active_query = """
    query GetClusterSpareActiveFiltered($targetTime: timestamptz!, $dcName: String!) {
      Metrics_Cluster_Metrics_Spare_Active(
        where: {time: {_eq: $targetTime}, datacenter: {name: {_eq: $dcName}}, cluster: {_nilike: "internal%", _neq: "Unknown"}}
      ) {
        cluster
        cpu_cores
        cpu_fragmented
        cpu_p95_pct
        cpu_reserved
        cpu_util_efect
        datacenter_id
        disk_capacity
        has_vms
        mem_capacity
        mem_p95_pct
        mem_reserved
        mem_reseverd_system
        mem_util_efect
        vcpu_capacity
        time
      }
    }
    """
    variables = {"dcName": site,"targetTime":time}
    
    try:
        query_result = mat_client.graphQL.execute(operation=cluster_spare_active_query,variables=variables)

        data = query_result.get("data", {}).get("Metrics_Cluster_Metrics_Spare_Active", [])
        job.log.info(f"Fetched {len(data)} Cluster spare/active Metrics.")
        
        return data
        
    except Exception as e:
        job.log.error(f"Failed to query metrics: {e}")
        raise e

def count_active_spare_servers(server_list):
    """
    Aggregates server list into a DataFrame with columns: 
    ['cluster', 'active_server_count', 'spare_server_count']
    """
    import pandas as pd
    if not server_list:
        return pd.DataFrame(columns=['cluster', 'active_server_count', 'spare_server_count'])

    rows = []
    for s in server_list:
        cluster = "Unknown"
        if s.get("cluster_server") and s["cluster_server"].get("name"):
            cluster = s["cluster_server"]["name"]
        
        role = s.get("role", "").lower()
        is_spare = "spare" in role
        
        rows.append({
            "cluster": cluster,
            "is_spare": is_spare
        })
    
    df = pd.DataFrame(rows)
    
    # Group by cluster and count active/spare
    # Active: is_spare == False
    # Spare: is_spare == True
    
    # Count Active
    active = df[~df['is_spare']].groupby('cluster').size().reset_index(name='active_server_count')
    
    # Count Spare
    spare = df[df['is_spare']].groupby('cluster').size().reset_index(name='spare_server_count')
    
    # Merge
    combined = pd.merge(active, spare, on='cluster', how='outer').fillna(0)
    
    return combined




def build_top10_idle_tables_vnfc(job, vnfc_metrics, dc_name):
    """
    vnfc_metrics: list[dict] como el que pegaste (cada item tiene vnf.name, resv_*, idle_*, efect_util_*_pct)
    dc_name: "Mayo" (solo para filtrar si querés)

    Devuelve un dict con 3 listas (list[dict]) listas para el template:
      - top10_idle_cpu
      - top10_idle_mem
      - top10_idle_disk
    """
    import pandas as pd

    def _vnf_type_from_name(v):
        # "PCRF01_Mayo" -> "PCRF01"
        if isinstance(v, str) and v:
            return v.split("_")[0]
        return "N/A"

    def _num(x):
        # Convierte a float si se puede; si no, NaN
        try:
            return float(x)
        except Exception:
            return None

    def _fmt(x, decimals=1, na="N/A"):
        if x is None:
            return na
        try:
            return f"{float(x):.{decimals}f}"
        except Exception:
            return na

    def _fmt_pct(x, decimals=2, na="N/A"):
        if x is None:
            return na
        try:
            return f"{float(x):.{decimals}f}"
        except Exception:
            return na

    df = pd.DataFrame(vnfc_metrics or [])

    if df.empty:
        return {"top10_idle_cpu": [], "top10_idle_mem": [], "top10_idle_disk": []}

    # Aplanar vnf.name y sacar tipo
    # Aplanar vnf.name y sacar tipo, y role si existe
    df["vnf_role"] = None
    if "vnf" in df.columns:
        df["vnf_name"] = df["vnf"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)
    elif "vnfc" in df.columns:
        # Fallback para estructura vnfc
        df["vnf_name"] = df["vnfc"].apply(lambda x: x.get("vnfc") if isinstance(x, dict) else None)
        df["vnf_role"] = df["vnfc"].apply(lambda x: x.get("role") if isinstance(x, dict) else None)
    else:
        df["vnf_name"] = None

    # Extract thresholds
    def _extract_thresholds(x):
        # x is the vnfc column (dict)
        defaults = {"high": 70, "low": 50} # Fail-safe defaults if not found
        if isinstance(x, dict) and "usage_thresholds" in x:
            return x["usage_thresholds"].get("high", 70), x["usage_thresholds"].get("low", 50)
        return 70, 50

    if "vnfc" in df.columns:
        df["thresholds"] = df["vnfc"].apply(_extract_thresholds)
    else:
        df["thresholds"] = [(70, 50)] * len(df)

    df["threshold_high"] = df["thresholds"].apply(lambda t: t[0])
    df["threshold_low"] = df["thresholds"].apply(lambda t: t[1])

    
    
    # Construct vnf_instance
    def _clean_name(name, role):
        if not isinstance(name, str):
            return ""
        
        cleaned = name
        
        # Remove Role if prefix (e.g. "PL_AAA..." with role "PL")
        # Check carefully to remove following underscore if present
        if role and isinstance(role, str):
            # Try removing "Role_" or "Role"
            if cleaned.startswith(f"{role}_"):
                 cleaned = cleaned[len(role)+1:]
            elif cleaned.startswith(role):
                 cleaned = cleaned[len(role):]
        
        # Replace underscores with spaces
        cleaned = cleaned.replace("_", " ")
        return cleaned.strip()

    df["vnf_instance"] = df.apply(
        lambda r: f"{_clean_name(r['vnf_name'], r['vnf_role'])} {r['vnf_role']}" if r["vnf_name"] and r["vnf_role"] else (_clean_name(r['vnf_name'], None) or "N/A"),
        axis=1
    )
    
    # Deduplicate vnf_instance
    # If duplicates exist (e.g. same name/role but different VNF), append suffix
    # We do this by iterating and tracking counts
    
    # Get values as list
    instances = df["vnf_instance"].tolist()
    seen = {}
    new_instances = []
    for inst in instances:
        if inst in seen:
            seen[inst] += 1
            new_instances.append(f"{inst} ({seen[inst]})")
        else:
            seen[inst] = 1
            new_instances.append(inst)
            
    df["vnf_instance"] = new_instances
    
    df["vnf_type"] = df["vnf_name"].apply(_vnf_type_from_name)

    # Normalizar columnas numéricas relevantes
    for c in [
        "resv_cpu", "idle_cpu", "idle_cpu_pct", "efect_util_cpu_pct",
        "resv_mem", "idle_mem", "idle_mem_pct", "efect_util_mem_pct",
        "resv_disk", "idle_disk", "idle_disk_pct", "efect_util_disk_pct",
        "mem_p95_raw", "cpu_p95_raw", "ram_p95", "cpu_p95"
    ]:
        if c in df.columns:
            df[c] = df[c].apply(_num)

    # --- CPU: calcular util efectiva "absoluta" si hace falta ---
    # Regla: util_cpu_abs = resv_cpu - idle_cpu  (si ambos están)
    df["util_cpu_abs"] = df.apply(
        lambda r: (r["resv_cpu"] - r["idle_cpu"])
        if (r.get("resv_cpu") is not None and r.get("idle_cpu") is not None)
        else None,
        axis=1
    )

    # --- MEM: util efectiva absoluta ---
    df["util_mem_abs"] = df.apply(
        lambda r: (r["resv_mem"] - r["idle_mem"])
        if (r.get("resv_mem") is not None and r.get("idle_mem") is not None)
        else None,
        axis=1
    )

    # --- DISK: util efectiva absoluta ---
    df["util_disk_abs"] = df.apply(
        lambda r: (r["resv_disk"] - r["idle_disk"])
        if (r.get("resv_disk") is not None and r.get("idle_disk") is not None)
        else None,
        axis=1
    )

    # Helpers para armar columnas tipo "985.7 (80.01%)"
    def _fmt_value_pct(value, pct, value_decimals=1, pct_decimals=2):
        if value is None and pct is None:
            return "N/A"
        if value is None:
            return f"N/A ({_fmt_pct(pct, pct_decimals)}\\%)"
        if pct is None:
            return f"{_fmt(value, value_decimals)} (N/A\\%)"
        return f"{_fmt(value, value_decimals)} ({_fmt_pct(pct, pct_decimals)}\\%)"

    # ======================
    # TOP10 CPU OCIOSA
    # ======================
    cpu_df = df.copy()
    # Orden: mayor idle_cpu (si None, al final)
    cpu_df = cpu_df.sort_values(by=["idle_cpu"], ascending=False, na_position="last").head(10)

    top10_idle_cpu = []
    for _, r in cpu_df.iterrows():
        top10_idle_cpu.append({
            "vnf_type": r.get("vnf_type") or "N/A",
            "resv_cpu": _fmt(r.get("resv_cpu"), decimals=0),
            "idle_cpu_str": _fmt_value_pct(r.get("idle_cpu"), r.get("idle_cpu_pct"), value_decimals=1, pct_decimals=2),
            "util_cpu_str": _fmt_value_pct(r.get("util_cpu_abs"), r.get("efect_util_cpu_pct"), value_decimals=1, pct_decimals=2),
            "util_local_cpu_str": _fmt_value_pct(
                r.get("cpu_p95_raw"),
                r.get("cpu_p95"),
                value_decimals=1,
                pct_decimals=2
            ),
            "vnf_instance": r.get("vnf_instance") or "N/A",
            "idle_cpu": r.get("idle_cpu"),
            "flavor:Vcpus": r.get("resv_cpu"),
            "cpu_p95_raw": r.get("cpu_p95_raw"),
            "cpu_p95": r.get("cpu_p95"),
            "threshold_high": r.get("threshold_high"),
            "threshold_low": r.get("threshold_low")
        })


    # ======================
    # TOP10 MEM OCIOSA
    # ======================
    mem_df = df.copy()
    mem_df = mem_df.sort_values(by=["idle_mem"], ascending=False, na_position="last").head(10)

    top10_idle_mem = []
    for _, r in mem_df.iterrows():
        top10_idle_mem.append({
            "vnf_type": r.get("vnf_type") or "N/A",
            "resv_mem": _fmt(
                r.get("resv_mem") / (1024 ** 3) if r.get("resv_mem") is not None else None,
                decimals=1
            ),
            "idle_mem_str": _fmt_value_pct(
                r.get("idle_mem") / (1024 ** 3) if r.get("idle_mem") is not None else None,
                r.get("idle_mem_pct"),
                value_decimals=1,
                pct_decimals=2
            ),
            "util_mem_str": _fmt_value_pct(
                r.get("util_mem_abs") / (1024 ** 3) if r.get("util_mem_abs") is not None else None,
                r.get("efect_util_mem_pct"),
                value_decimals=1,
                pct_decimals=2
            ),
            "util_local_mem_str": _fmt_value_pct(
                r.get("mem_p95_raw") / (1024 ** 3) if r.get("mem_p95_raw") is not None else None,
                r.get("ram_p95"),
                value_decimals=1,
                pct_decimals=2
            ),
            "vnf_instance": r.get("vnf_instance") or "N/A",
            "idle_mem": r.get("idle_mem") / (1024 ** 3) if r.get("idle_mem") is not None else None,
            "flavor:Ram": r.get("resv_mem") / (1024 ** 3) if r.get("resv_mem") is not None else None,
            "mem_p95_raw": r.get("mem_p95_raw"),
            "ram_p95": r.get("ram_p95"),
            "threshold_high": r.get("threshold_high"),
            "threshold_low": r.get("threshold_low")
        })


    # ======================
    # TOP10 DISK OCIOSO
    # ======================
    disk_df = df.copy()
    disk_df = disk_df.sort_values(by=["idle_disk"], ascending=False, na_position="last").head(10)

    top10_idle_disk = []
    for _, r in disk_df.iterrows():
        top10_idle_disk.append({
            "vnf_type": r.get("vnf_type") or "N/A",
            "resv_disk": _fmt(
                r.get("resv_disk") / (1024) if r.get("resv_disk") is not None else None,
                decimals=1
            ),
            "idle_disk_str": _fmt_value_pct(
                r.get("idle_disk") / (1024) if r.get("idle_disk") is not None else None,
                r.get("idle_disk_pct"),
                value_decimals=1,
                pct_decimals=2
            ),
            "util_disk_str": _fmt_value_pct(
                r.get("util_disk_abs") / (1024) if r.get("util_disk_abs") is not None else None,
                r.get("efect_util_disk_pct"),
                value_decimals=1,
                pct_decimals=2
            ),
            "vnf_instance": r.get("vnf_instance") or "N/A",
            "idle_disk": r.get("idle_disk") / (1024) if r.get("idle_disk") is not None else None,
            "flavor:Disk": r.get("resv_disk") / (1024) if r.get("resv_disk") is not None else None,
            "threshold_high": r.get("threshold_high"),
            "threshold_low": r.get("threshold_low")
        })


    return {
        "top10_idle_cpu": top10_idle_cpu,
        "top10_idle_mem": top10_idle_mem,
        "top10_idle_disk": top10_idle_disk,
    }


def build_top10_cpu_p95(data):
    
    import pandas as pd
    
    df = pd.DataFrame(data)
    
    # Aplanar datacenter desde la columna vnf
    df["name"] = df["vnf"].apply(
        lambda x: x.get("name").split("_")[0]
        if isinstance(x, dict) and x.get("name")
        else None
    )

    df["datacenter"] = df["vnf"].apply(
        lambda x: x.get("datacenter").replace("_", " ")
        if isinstance(x, dict) and x.get("datacenter")
        else None
    )

    # Filtrar cpu_p95 válidos
    df = df[df["cpu_p95"].notna()]

    # Ordenar descendente y tomar top 5
    top5 = (
        df.sort_values("cpu_p95", ascending=False)
          .head(10)
          .reset_index(drop=True)
    )

    return top5[[
        "vnf_id",
        "name",
        "datacenter",
        "cpu_p95",
        "ram_p95",
        "efect_util_cpu_pct",
        "efect_util_mem_pct",
        "efect_util_disk_pct"
    ]]
    

def get_dc_monthly(region,mat_client):
    
    query="""query get_dc_metrics($region: String!) {
  Metrics_DC_Metrics_Monthly(
    distinct_on: [datacenter_id]
    order_by: [{datacenter_id: asc}, {time: desc}]
    where: {datacenter: {region: {_eq: $region}}}
  ) {
    datacenter{name}
    efect_util_cpu
    cpu_p95_pct
    efect_util_mem
    mem_p95_pct
    efect_util_disk_pct
  }
}"""

    query_result = mat_client.graphQL.execute(operation=query, variables={"region":region})
    query_data = query_result.get("data", {}).get("Metrics_DC_Metrics_Monthly", [])

    return query_data