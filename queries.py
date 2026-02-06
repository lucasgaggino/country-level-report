
GET_REGION_SITES = """
query get_sites($region: String, $limit: Int!, $offset: Int) {
  Resources_Datacenter(
    where: {_and: [{region: {_eq: $region}}, {site: {_nin: ["Morales", "Portales"]}}]}
    limit: $limit
    offset: $offset
  ) {
    site
    region
  }
}
"""


GET_DC_METRICS = """
query get_sites_metrics($sites: [String!]!, $since: timestamptz!, $limit: Int!, $offset: Int) {
  Metrics_DC_Metrics_Monthly(
    where: {_and: [{datacenter: {name: {_in: $sites}}}, {time: {_gt: $since}}]}
    order_by: [{datacenter_id: asc}, {time: desc}]
    distinct_on: [datacenter_id]
    limit: $limit
    offset: $offset
  ) {
    datacenter {
      name
    }
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
    cpu_p95_pct
    mem_p95_pct
    region
    time
    cap_neta_cpu
    cap_neta_mem
    cap_neta_disk
  }
}
"""

GET_REGION_METRICS = """
query get_sites_metrics($region: String!, $since: timestamptz!, $limit: Int!, $offset: Int) {
  Metrics_Region_Metrics_Monthly(
    where: {_and: [{region: {_eq: $region}}, {time: {_gt: $since}}]}
    order_by: [{region: asc}, {time: desc}]
    distinct_on: [region]
    limit: $limit
    offset: $offset
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
    cpu_p95_pct
    mem_p95_pct
    region
    time
    cap_neta_cpu
    cap_neta_mem
    cap_neta_disk
  }
}
"""

GET_CLUSTER_METRICS = """
query GetClusterMetricsDistinct($sites: [String!]!, $since: timestamptz!, $limit: Int!, $offset: Int) {
  Metrics_Cluster_Metrics_Monthly(
    distinct_on: [cluster]
    order_by: [{cluster: asc}, {time: desc}]
    where: {_and: [{datacenter: {name: {_in: $sites}}}, {time: {_gt: $since}}]}
    limit: $limit
    offset: $offset
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


GET_ROLES = """
query GetRolesByClusterAndSite($names: [String!], $sites: [String!], $limit: Int!, $offset: Int) {
  Resources_Cluster(
    where: {clusterId: {_in: $names}, datacenter_cluster: {name: {_in: $sites}}}
    limit: $limit
    offset: $offset
  ) {
    clusterId
    role
  }
}
"""


GET_SERVER = """
query GetServerMetrics($sites: [String!], $limit: Int!, $offset: Int) {
  Resources_Server(
    where: {rack_server: {dc_rack: {name: {_in: $sites}}}}
    limit: $limit
    offset: $offset
  ) {
    rack_server {
      dc_rack {
        name
      }
    }
    hostname
    role
    cluster_server {
      name
      clusterId
    }
    storage_total_gb
  }
}
"""


GET_VNF_TYPES = """
query GetVNFTypes($sites: [String!], $limit: Int!, $offset: Int) {
  Resources_VNF(
    where: {vnf_instancevnf: {datacenter: {_in: $sites}}}
    limit: $limit
    offset: $offset
  ) {
    vnf_instancevnf {
      datacenter
    }
    vnf
  }
}
"""


GET_VNF_METRICS = """
query GetVNFMetricsDistinct($sites: [String!], $since: timestamptz!, $limit: Int!, $offset: Int) {
  Metrics_VNF_Metrics_Monthly(
    where: {_and: [{vnf: {datacenter: {_in: $sites}}}, {time: {_gt: $since}}]}
    order_by: [{vnf_id: asc}, {time: desc}]
    distinct_on: [vnf_id]
    limit: $limit
    offset: $offset
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


GET_VMS = """
query GetVMInfo($sites: [String!], $limit: Int!, $offset: Int) {
  Resources_VM(
    where: {vnfc_vm: {vnfc_instancevnf: {datacenter: {_in: $sites}}}}
    limit: $limit
    offset: $offset
  ) {
    name
    vm
    hypervisor_vm {
      cluster
    }
    vnfc_vm {
      vnfc_instancevnf {
        datacenter
        vnf_instancevnf {
          name
        }
      }
    }
  }
}
"""

GET_VNFC = """
query GetVNFCMetricsByInstanceVNF($sites: [String!], $since: timestamptz!, $limit: Int!, $offset: Int!){
  Metrics_VNFC_Metrics_Monthly(
    distinct_on: [vnfc_id]
    order_by: [{vnfc_id: asc}, {time: desc}]
    where: {_and: [{vnfc: {vnfc_instancevnf: {datacenter: {_in: $sites}}}}, {time: {_gt: $since}}]}
    limit: $limit
    offset: $offset
  ) {
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
    mem_p95_pct
    cpu_p95_pct
    vnfc {
      vnfc
      role
      usage_thresholds {
        high
        low
      }
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


GET_NAS = """
query GetNASMetrics($sites: [String!], $limit: Int!, $offset: Int!) {
  Resources_Storage_Pool(
    where: {pools: {node_a: {rack_server: {dc: {_in: $sites}}}}}
    limit: $limit
    offset: $offset
  ) {
    pools {
      node_a {
        rack_server {
          dc
        }
      }
    }
    capacity_total_tib
    capacity_used_tib
  }
}
"""

GET_VM_DISK_METRICS = """
query GetLatestMetricsPerDisk($sites: [String!], $since: timestamptz!, $limit: Int!, $offset: Int!) {
  Metrics_VM_disk_metrics(
    distinct_on: [vm_id, disk]
    order_by: [{vm_id: asc}, {disk: asc}, {time: desc}]
    where: {_and: [{vm: {vnfc_vm: {vnfc_instancevnf: {datacenter: {_in: $sites}}}}}, {time: {_gt: $since}}]}
    limit: $limit
    offset: $offset
  ) {
    time
    disk
    used
    total
    vm {
      vnfc_vm {
        vnfc_instancevnf {
          datacenter
        }
      }
      name
      hypervisor_vm {
        name
      }
    }
  }
}
"""


GET_SERVER_DISK_METRICS = """
query GetLatestMetricsPerDisk($sites: [String!], $since: timestamptz!, $limit: Int!, $offset: Int!) {
  Metrics_Server_disk_Metrics(
    distinct_on: [server_id]
    order_by: [{server_id: asc}, {time: desc}]
    where: {_and: [{server: {rack_server: {dc: {_in: $sites}}}}, {time: {_gt: $since}}]}
    limit: $limit
    offset: $offset
  ) {
    server {
      rack_server {
        dc
      }
      hostname
      server
      mat_pk
      storage_total_gb
      cluster_server {
        name
        clusterId
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

GET_CLUSTER_SPARE_ACTIVE_METRICS = """
query GetClusterSpareActiveFiltered($sites: [String!], $since: timestamptz!, $limit: Int!, $offset: Int!) {
  Metrics_Cluster_Metrics_Spare_Active(
    where: {_and: [{datacenter: {name: {_in: $sites}}}, {cluster: {_nilike: "internal%", _neq: "Unknown"}}, {time: {_gt: $since}}]}
    limit: $limit
    offset: $offset
  ) {
    datacenter {
      name
    }
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

GET_REDUNDANCY =  """
query GetRedundancyInfo2($region: String!, $limit: Int!, $offset: Int!) {
  Resources_VNFc_Redundancy(
    where: {groupName: {_ilike: $region}}
    limit: $limit
    offset: $offset
  ) {
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