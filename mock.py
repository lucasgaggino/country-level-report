region = "9"

dc_list_str = "San Juan, Nextengo Nacional, Nextengo Regional, Popotla"

overview_list= [
    {
        "dc": "San Juan",
        "clusters": "14",
        "total_servers": "246",
        "spare_servers": "88",
        "spare_pct": "35.77",
        "vnf_types": "17",
        "vms": "455"
    },
    {
        "dc": "Nextengo Nacional",
        "clusters": "12",
        "total_servers": "198",
        "spare_servers": "72",
        "spare_pct": "36.36",
        "vnf_types": "15",
        "vms": "380"
    },
    {
        "dc": "Nextengo Regional",
        "clusters": "10",
        "total_servers": "164",
        "spare_servers": "58",
        "spare_pct": "35.37",
        "vnf_types": "14",
        "vms": "312"
    },
    {
        "dc": "Popotla",
        "clusters": "8",
        "total_servers": "132",
        "spare_servers": "48",
        "spare_pct": "36.36",
        "vnf_types": "12",
        "vms": "248"
    }
]
        
total_overview = {
    "dc": f'Total Región {region}',
    "clusters": f'{sum([int(item.get("clusters",0)) for item in overview_list])}',
    "total_servers": f'{sum([int(item.get("total_servers",0)) for item in overview_list])}',
    "spare_servers": f'{sum([int(item.get("spare_servers",0)) for item in overview_list])}',
    "spare_pct": 'null',
    "vnf_types": 'null',
    "vms": f'{sum([int(item.get("vms",0)) for item in overview_list])}'
}

region_metrics = [
    {
        "dc": "San Juan",
        "cpu_net_cap": 13000,
        "cpu_eff_res": 50.15,
        "cpu_eff_util": 63.21,
        "mem_net_cap": 56.45,
        "mem_eff_res": 31.16,
        "mem_eff_util": 61.40,
        "disk_net_cap": 287.01,
        "disk_eff_res": 9.65,
        "disk_eff_util": 67.50
    },
    {
        "dc": "Nextengo Nacional",
        "cpu_net_cap": 10500,
        "cpu_eff_res": 48.22,
        "cpu_eff_util": 58.45,
        "mem_net_cap": 45.20,
        "mem_eff_res": 29.80,
        "mem_eff_util": 55.30,
        "disk_net_cap": 230.50,
        "disk_eff_res": 8.90,
        "disk_eff_util": 62.15
    },
    {
        "dc": "Nextengo Regional",
        "cpu_net_cap": 8200,
        "cpu_eff_res": 45.80,
        "cpu_eff_util": 52.30,
        "mem_net_cap": 35.60,
        "mem_eff_res": 27.50,
        "mem_eff_util": 48.90,
        "disk_net_cap": 180.25,
        "disk_eff_res": 7.80,
        "disk_eff_util": 58.40
    },
    {
        "dc": "Popotla",
        "cpu_net_cap": 6600,
        "cpu_eff_res": 42.50,
        "cpu_eff_util": 48.75,
        "mem_net_cap": 28.40,
        "mem_eff_res": 25.20,
        "mem_eff_util": 45.60,
        "disk_net_cap": 145.00,
        "disk_eff_res": 6.95,
        "disk_eff_util": 54.20
    }
]

total_region_metric = {
    "dc": "Total Region 9",
    "cpu_net_cap": 38300,
    "cpu_eff_res": 46.67,
    "cpu_eff_util": 55.68,
    "mem_net_cap": 165.65,
    "mem_eff_res": 28.42,
    "mem_eff_util": 52.80,
    "disk_net_cap": 842.76,
    "disk_eff_res": 8.33,
    "disk_eff_util": 60.56
}

nas_storages = [
    {
        "dc": "San Juan",
        "total_cap": 49.16,
        "util": 2.05,
        "util_pct": 4.17
    },
    {
        "dc": "Nextengo Nacional",
        "total_cap": 38.50,
        "util": 1.82,
        "util_pct": 4.73
    },
    {
        "dc": "Nextengo Regional",
        "total_cap": 32.00,
        "util": 1.45,
        "util_pct": 4.53
    },
    {
        "dc": "Popotla",
        "total_cap": 25.80,
        "util": 1.12,
        "util_pct": 4.34
    }
]

total_nas_storage = {
    "dc": "Total Region 9",
    "total_cap": 145.46,
    "util": 6.44,
    "util_pct": 4.43
}

vm_utilization = [
    {
        "dc": "San Juan",
        "cpu_eff_util_pct": 63.21,
        "mem_eff_util_pct": 61.40,
        "cpu_local_util_pct": 48.52,
        "mem_local_util_pct": 52.18,
        "disk_util_pct": 67.50
    },
    {
        "dc": "Nextengo Nacional",
        "cpu_eff_util_pct": 58.45,
        "mem_eff_util_pct": 55.30,
        "cpu_local_util_pct": 45.20,
        "mem_local_util_pct": 48.65,
        "disk_util_pct": 62.15
    },
    {
        "dc": "Nextengo Regional",
        "cpu_eff_util_pct": 52.30,
        "mem_eff_util_pct": 48.90,
        "cpu_local_util_pct": 40.15,
        "mem_local_util_pct": 42.80,
        "disk_util_pct": 58.40
    },
    {
        "dc": "Popotla",
        "cpu_eff_util_pct": 48.75,
        "mem_eff_util_pct": 45.60,
        "cpu_local_util_pct": 36.90,
        "mem_local_util_pct": 39.25,
        "disk_util_pct": 54.20
    }
]

avg_vm_utilization = {
    "dc": "Promedio Region 9",
    "cpu_eff_util_pct": 55.68,
    "mem_eff_util_pct": 52.80,
    "cpu_local_util_pct": 42.69,
    "mem_local_util_pct": 45.72,
    "disk_util_pct": 60.56
}

top5_vnf_cpu_p95_region = [
    {
        "vnf_instance": "PGWU01",
        "dc": "San Juan",
        "cpu_local_p95_pct": 95.94,
        "cpu_eff_p95_pct": 129.18,
        "ram_local_p95_pct": 45.20,
        "ram_eff_p95_pct": 54.08,
        "disk_util_pct": 54.37
    },
    {
        "vnf_instance": "PGWC01",
        "dc": "Nextengo Nacional",
        "cpu_local_p95_pct": 95.47,
        "cpu_eff_p95_pct": 118.34,
        "ram_local_p95_pct": 36.04,
        "ram_eff_p95_pct": 42.98,
        "disk_util_pct": 73.01
    },
    {
        "vnf_instance": "DRA01",
        "dc": "Popotla",
        "cpu_local_p95_pct": 66.87,
        "cpu_eff_p95_pct": 109.98,
        "ram_local_p95_pct": 33.14,
        "ram_eff_p95_pct": 46.28,
        "disk_util_pct": 78.83
    },
    {
        "vnf_instance": "MME01",
        "dc": "Nextengo Regional",
        "cpu_local_p95_pct": 63.15,
        "cpu_eff_p95_pct": 85.13,
        "ram_local_p95_pct": 14.88,
        "ram_eff_p95_pct": 17.74,
        "disk_util_pct": 79.16
    },
    {
        "vnf_instance": "CSCF02",
        "dc": "San Juan",
        "cpu_local_p95_pct": 61.16,
        "cpu_eff_p95_pct": 103.08,
        "ram_local_p95_pct": 79.31,
        "ram_eff_p95_pct": 117.86,
        "disk_util_pct": 100.00
    }
]


vnf_vmtype_utilization = [
    {"vnf_instance": "AAA01", "vm_type": "PL", "cpu_eff_pct": 53.61, "mem_eff_pct": 87.24, "cpu_local_pct": 24.68, "mem_local_pct": 48.24, "disk_util_pct": 49.36, "vm_count": 4},
    {"vnf_instance": "AAA01", "vm_type": "SC", "cpu_eff_pct": 121.64, "mem_eff_pct": 155.47, "cpu_local_pct": 52.85, "mem_local_pct": 84.00, "disk_util_pct": 83.31, "vm_count": 2},
    {"vnf_instance": "BGF01", "vm_type": "SC+PL", "cpu_eff_pct": 86.61, "mem_eff_pct": 64.09, "cpu_local_pct": 50.91, "mem_local_pct": 45.37, "disk_util_pct": 22.67, "vm_count": 24},
    {"vnf_instance": "CSCF01", "vm_type": "PL", "cpu_eff_pct": 105.54, "mem_eff_pct": 121.55, "cpu_local_pct": 61.68, "mem_local_pct": 81.58, "disk_util_pct": 49.11, "vm_count": 18},
    {"vnf_instance": "CSCF02", "vm_type": "PL", "cpu_eff_pct": 107.90, "mem_eff_pct": 121.74, "cpu_local_pct": 64.04, "mem_local_pct": 81.78, "disk_util_pct": 49.11, "vm_count": 18},
    {"vnf_instance": "CSCF03", "vm_type": "PL", "cpu_eff_pct": 50.01, "mem_eff_pct": 113.67, "cpu_local_pct": 6.15, "mem_local_pct": 73.70, "disk_util_pct": 47.80, "vm_count": 3},
    {"vnf_instance": "CSCF01", "vm_type": "SC", "cpu_eff_pct": 22.97, "mem_eff_pct": 29.10, "cpu_local_pct": 15.90, "mem_local_pct": 23.10, "disk_util_pct": 100.00, "vm_count": 2},
    {"vnf_instance": "CSCF02", "vm_type": "SC", "cpu_eff_pct": 16.24, "mem_eff_pct": 28.40, "cpu_local_pct": 9.17, "mem_local_pct": 22.39, "disk_util_pct": 100.00, "vm_count": 2},
    {"vnf_instance": "CSCF03", "vm_type": "SC", "cpu_eff_pct": 9.82, "mem_eff_pct": 26.24, "cpu_local_pct": 2.75, "mem_local_pct": 20.23, "disk_util_pct": 100.00, "vm_count": 2},
    {"vnf_instance": "CUDB11", "vm_type": "PL", "cpu_eff_pct": 11.16, "mem_eff_pct": 127.83, "cpu_local_pct": 7.08, "mem_local_pct": 91.33, "disk_util_pct": 66.61, "vm_count": 18},
    {"vnf_instance": "CUDB11", "vm_type": "SC", "cpu_eff_pct": 33.90, "mem_eff_pct": 25.73, "cpu_local_pct": 26.42, "mem_local_pct": 19.15, "disk_util_pct": 68.03, "vm_count": 2},
    {"vnf_instance": "DRA01", "vm_type": "PL", "cpu_eff_pct": 110.83, "mem_eff_pct": 46.36, "cpu_local_pct": 67.44, "mem_local_pct": 33.21, "disk_util_pct": 0.08, "vm_count": 31},
    {"vnf_instance": "DRA01", "vm_type": "SC", "cpu_eff_pct": 83.60, "mem_eff_pct": 44.58, "cpu_local_pct": 49.11, "mem_local_pct": 31.64, "disk_util_pct": 91.04, "vm_count": 2},
    {"vnf_instance": "EDA11", "vm_type": "PG", "cpu_eff_pct": 19.98, "mem_eff_pct": 80.60, "cpu_local_pct": 19.98, "mem_local_pct": 80.60, "disk_util_pct": 1.84, "vm_count": 5},
    {"vnf_instance": "EPDG01", "vm_type": "VRP", "cpu_eff_pct": 25.52, "mem_eff_pct": 53.29, "cpu_local_pct": 16.48, "mem_local_pct": 34.58, "disk_util_pct": 55.06, "vm_count": 2},
    {"vnf_instance": "EPDG01", "vm_type": "VSFO", "cpu_eff_pct": 98.71, "mem_eff_pct": 79.98, "cpu_local_pct": 58.79, "mem_local_pct": 56.22, "disk_util_pct": 21.49, "vm_count": 20},
    {"vnf_instance": "IPWorks01", "vm_type": "PL", "cpu_eff_pct": 12.00, "mem_eff_pct": 74.81, "cpu_local_pct": 7.33, "mem_local_pct": 53.77, "disk_util_pct": 49.36, "vm_count": 5},
    {"vnf_instance": "IPWorks01", "vm_type": "SC", "cpu_eff_pct": 19.55, "mem_eff_pct": 125.92, "cpu_local_pct": 12.95, "mem_local_pct": 89.97, "disk_util_pct": 84.47, "vm_count": 2},
    {"vnf_instance": "JS-VM0", "vm_type": "JS-VM", "cpu_eff_pct": 1.57, "mem_eff_pct": 8.11, "cpu_local_pct": 1.57, "mem_local_pct": 8.11, "disk_util_pct": 21.25, "vm_count": 1},
    {"vnf_instance": "MME01", "vm_type": "FSB", "cpu_eff_pct": 23.17, "mem_eff_pct": 67.49, "cpu_local_pct": 16.52, "mem_local_pct": 56.60, "disk_util_pct": 86.33, "vm_count": 2},
    {"vnf_instance": "MME01", "vm_type": "GPB", "cpu_eff_pct": 86.76, "mem_eff_pct": 15.72, "cpu_local_pct": 64.38, "mem_local_pct": 13.20, "disk_util_pct": 0.08, "vm_count": 27},
    {"vnf_instance": "MME01", "vm_type": "NCB", "cpu_eff_pct": 22.54, "mem_eff_pct": 62.19, "cpu_local_pct": 15.96, "mem_local_pct": 51.84, "disk_util_pct": 0.08, "vm_count": 2},
    {"vnf_instance": "MRF01", "vm_type": "SC+PL", "cpu_eff_pct": 27.63, "mem_eff_pct": 84.65, "cpu_local_pct": 16.26, "mem_local_pct": 61.28, "disk_util_pct": 70.56, "vm_count": 7},
    {"vnf_instance": "MTAS01", "vm_type": "PL", "cpu_eff_pct": 64.96, "mem_eff_pct": 111.74, "cpu_local_pct": 39.00, "mem_local_pct": 80.15, "disk_util_pct": 49.37, "vm_count": 25},
    {"vnf_instance": "MTAS02", "vm_type": "PL", "cpu_eff_pct": 63.35, "mem_eff_pct": 111.65, "cpu_local_pct": 37.39, "mem_local_pct": 80.07, "disk_util_pct": 49.37, "vm_count": 25},
    {"vnf_instance": "MTAS01", "vm_type": "SC", "cpu_eff_pct": 19.07, "mem_eff_pct": 24.95, "cpu_local_pct": 11.99, "mem_local_pct": 17.85, "disk_util_pct": 100.00, "vm_count": 2},
    {"vnf_instance": "MTAS02", "vm_type": "SC", "cpu_eff_pct": 19.81, "mem_eff_pct": 24.87, "cpu_local_pct": 12.74, "mem_local_pct": 17.76, "disk_util_pct": 100.00, "vm_count": 2},
    {"vnf_instance": "PCRF01", "vm_type": "PL", "cpu_eff_pct": 8.84, "mem_eff_pct": 22.67, "cpu_local_pct": 8.84, "mem_local_pct": 22.67, "disk_util_pct": 49.36, "vm_count": 37},
    {"vnf_instance": "PCRF02", "vm_type": "PL", "cpu_eff_pct": 8.90, "mem_eff_pct": 16.49, "cpu_local_pct": 8.90, "mem_local_pct": 16.49, "disk_util_pct": 49.36, "vm_count": 37},
    {"vnf_instance": "PCRF01", "vm_type": "SC", "cpu_eff_pct": 50.86, "mem_eff_pct": 25.74, "cpu_local_pct": 50.86, "mem_local_pct": 25.74, "disk_util_pct": 55.78, "vm_count": 2},
    {"vnf_instance": "PCRF02", "vm_type": "SC", "cpu_eff_pct": 51.33, "mem_eff_pct": 28.77, "cpu_local_pct": 51.33, "mem_local_pct": 28.77, "disk_util_pct": 55.48, "vm_count": 2},
    {"vnf_instance": "PDRA01", "vm_type": "PL", "cpu_eff_pct": 58.91, "mem_eff_pct": 37.33, "cpu_local_pct": 48.51, "mem_local_pct": 27.17, "disk_util_pct": 0.08, "vm_count": 22},
    {"vnf_instance": "PDRA02", "vm_type": "PL", "cpu_eff_pct": 13.42, "mem_eff_pct": 34.55, "cpu_local_pct": 3.02, "mem_local_pct": 24.39, "disk_util_pct": 0.08, "vm_count": 22},
    {"vnf_instance": "PDRA01", "vm_type": "SC", "cpu_eff_pct": 61.04, "mem_eff_pct": 42.40, "cpu_local_pct": 46.19, "mem_local_pct": 31.75, "disk_util_pct": 90.74, "vm_count": 2},
    {"vnf_instance": "PDRA02", "vm_type": "SC", "cpu_eff_pct": 32.98, "mem_eff_pct": 38.30, "cpu_local_pct": 18.14, "mem_local_pct": 27.65, "disk_util_pct": 89.58, "vm_count": 2},
    {"vnf_instance": "PGWC01", "vm_type": "VRP", "cpu_eff_pct": 63.44, "mem_eff_pct": 49.34, "cpu_local_pct": 51.65, "mem_local_pct": 41.45, "disk_util_pct": 66.61, "vm_count": 2},
    {"vnf_instance": "PGWC01", "vm_type": "VSFO", "cpu_eff_pct": 119.42, "mem_eff_pct": 42.85, "cpu_local_pct": 96.32, "mem_local_pct": 35.92, "disk_util_pct": 74.08, "vm_count": 12},
    {"vnf_instance": "PGWU01", "vm_type": "VRP", "cpu_eff_pct": 81.52, "mem_eff_pct": 31.65, "cpu_local_pct": 61.11, "mem_local_pct": 27.10, "disk_util_pct": 66.82, "vm_count": 2},
    {"vnf_instance": "PGWU01", "vm_type": "VSFO", "cpu_eff_pct": 129.55, "mem_eff_pct": 54.64, "cpu_local_pct": 96.21, "mem_local_pct": 45.66, "disk_util_pct": 53.54, "vm_count": 30},
    {"vnf_instance": "SBG01", "vm_type": "PL", "cpu_eff_pct": 91.89, "mem_eff_pct": 78.47, "cpu_local_pct": 53.02, "mem_local_pct": 59.35, "disk_util_pct": 97.33, "vm_count": 26},
    {"vnf_instance": "SBG01", "vm_type": "SC", "cpu_eff_pct": 55.54, "mem_eff_pct": 41.25, "cpu_local_pct": 33.88, "mem_local_pct": 27.27, "disk_util_pct": 100.00, "vm_count": 2}
]

top10_idle_cpu_region = [
    {
        "vnf_instance": "PCRF01",
        "dc": "San Juan",
        "vcpu_reserved": 1266,
        "vcpu_idle": 1150.8,
        "vcpu_idle_pct": 90.90,
        "cpu_eff": 115.2,
        "cpu_eff_pct": 9.10,
        "cpu_local": 115.2,
        "cpu_local_pct": 9.10
    },
    {
        "vnf_instance": "PCRF02",
        "dc": "Nextengo Nacional",
        "vcpu_reserved": 1266,
        "vcpu_idle": 1150.0,
        "vcpu_idle_pct": 90.84,
        "cpu_eff": 116.0,
        "cpu_eff_pct": 9.16,
        "cpu_local": 116.0,
        "cpu_local_pct": 9.16
    },
    {
        "vnf_instance": "CUDB11",
        "dc": "Popotla",
        "vcpu_reserved": 320,
        "vcpu_idle": 277.0,
        "vcpu_idle_pct": 86.57,
        "cpu_eff": 43.0,
        "cpu_eff_pct": 13.43,
        "cpu_local": 28.9,
        "cpu_local_pct": 9.02
    },
    {
        "vnf_instance": "PDRA02",
        "dc": "Nextengo Regional",
        "vcpu_reserved": 276,
        "vcpu_idle": 236.6,
        "vcpu_idle_pct": 85.73,
        "cpu_eff": 39.4,
        "cpu_eff_pct": 14.27,
        "cpu_local": 10.1,
        "cpu_local_pct": 3.68
    },
    {
        "vnf_instance": "MRF01",
        "dc": "San Juan",
        "vcpu_reserved": 238,
        "vcpu_idle": 172.2,
        "vcpu_idle_pct": 72.37,
        "cpu_eff": 65.8,
        "cpu_eff_pct": 27.63,
        "cpu_local": 38.7,
        "cpu_local_pct": 16.26
    },
    {
        "vnf_instance": "MTAS02",
        "dc": "Nextengo Nacional",
        "vcpu_reserved": 420,
        "vcpu_idle": 162.6,
        "vcpu_idle_pct": 38.73,
        "cpu_eff": 257.4,
        "cpu_eff_pct": 61.27,
        "cpu_local": 152.1,
        "cpu_local_pct": 36.21
    },
    {
        "vnf_instance": "MTAS01",
        "dc": "Popotla",
        "vcpu_reserved": 420,
        "vcpu_idle": 156.3,
        "vcpu_idle_pct": 37.23,
        "cpu_eff": 263.7,
        "cpu_eff_pct": 62.77,
        "cpu_local": 158.4,
        "cpu_local_pct": 37.71
    },
    {
        "vnf_instance": "MME01",
        "dc": "San Juan",
        "vcpu_reserved": 942,
        "vcpu_idle": 140.1,
        "vcpu_idle_pct": 14.87,
        "cpu_eff": 801.9,
        "cpu_eff_pct": 85.13,
        "cpu_local": 594.9,
        "cpu_local_pct": 63.15
    },
    {
        "vnf_instance": "PDRA01",
        "dc": "Nextengo Regional",
        "vcpu_reserved": 276,
        "vcpu_idle": 113.2,
        "vcpu_idle_pct": 41.00,
        "cpu_eff": 162.8,
        "cpu_eff_pct": 59.00,
        "cpu_local": 133.6,
        "cpu_local_pct": 48.41
    },
    {
        "vnf_instance": "IPWorks01",
        "dc": "Popotla",
        "vcpu_reserved": 98,
        "vcpu_idle": 84.1,
        "vcpu_idle_pct": 85.84,
        "cpu_eff": 13.9,
        "cpu_eff_pct": 14.16,
        "cpu_local": 8.8,
        "cpu_local_pct": 8.93
    }
]

top10_idle_mem_region = [
    {
        "vnf_instance": "MME01",
        "dc": "San Juan",
        "ram_reserved_gb": 2820.0,
        "ram_idle_gb": 2319.9,
        "ram_idle_pct": 82.26,
        "mem_eff_gb": 500.1,
        "mem_eff_pct": 17.74,
        "mem_local_gb": 419.6,
        "mem_local_pct": 14.88
    },
    {
        "vnf_instance": "PCRF02",
        "dc": "Nextengo Nacional",
        "ram_reserved_gb": 2392.0,
        "ram_idle_gb": 1994.7,
        "ram_idle_pct": 83.39,
        "mem_eff_gb": 397.3,
        "mem_eff_pct": 16.61,
        "mem_local_gb": 397.3,
        "mem_local_pct": 16.61
    },
    {
        "vnf_instance": "PCRF01",
        "dc": "San Juan",
        "ram_reserved_gb": 2392.0,
        "ram_idle_gb": 1849.0,
        "ram_idle_pct": 77.30,
        "mem_eff_gb": 543.0,
        "mem_eff_pct": 22.70,
        "mem_local_gb": 543.0,
        "mem_local_pct": 22.70
    },
    {
        "vnf_instance": "PGWU01",
        "dc": "Popotla",
        "ram_reserved_gb": 1968.0,
        "ram_idle_gb": 903.6,
        "ram_idle_pct": 45.92,
        "mem_eff_gb": 1064.4,
        "mem_eff_pct": 54.08,
        "mem_local_gb": 889.6,
        "mem_local_pct": 45.20
    },
    {
        "vnf_instance": "DRA01",
        "dc": "Nextengo Regional",
        "ram_reserved_gb": 1040.0,
        "ram_idle_gb": 558.7,
        "ram_idle_pct": 53.72,
        "mem_eff_gb": 481.3,
        "mem_eff_pct": 46.28,
        "mem_local_gb": 344.6,
        "mem_local_pct": 33.14
    },
    {
        "vnf_instance": "PDRA02",
        "dc": "San Juan",
        "ram_reserved_gb": 752.0,
        "ram_idle_gb": 490.4,
        "ram_idle_pct": 65.21,
        "mem_eff_gb": 261.6,
        "mem_eff_pct": 34.79,
        "mem_local_gb": 185.0,
        "mem_local_pct": 24.60
    },
    {
        "vnf_instance": "PDRA01",
        "dc": "Nextengo Nacional",
        "ram_reserved_gb": 752.0,
        "ram_idle_gb": 468.8,
        "ram_idle_pct": 62.34,
        "mem_eff_gb": 283.2,
        "mem_eff_pct": 37.66,
        "mem_local_gb": 206.5,
        "mem_local_pct": 27.46
    },
    {
        "vnf_instance": "SBG01",
        "dc": "Popotla",
        "ram_reserved_gb": 2096.0,
        "ram_idle_gb": 457.2,
        "ram_idle_pct": 21.82,
        "mem_eff_gb": 1638.8,
        "mem_eff_pct": 78.18,
        "mem_local_gb": 1238.8,
        "mem_local_pct": 59.10
    },
    {
        "vnf_instance": "PGWC01",
        "dc": "Nextengo Regional",
        "ram_reserved_gb": 784.0,
        "ram_idle_gb": 447.0,
        "ram_idle_pct": 57.02,
        "mem_eff_gb": 337.0,
        "mem_eff_pct": 42.98,
        "mem_local_gb": 282.5,
        "mem_local_pct": 36.04
    },
    {
        "vnf_instance": "EPDG01",
        "dc": "San Juan",
        "ram_reserved_gb": 1136.0,
        "ram_idle_gb": 231.7,
        "ram_idle_pct": 20.40,
        "mem_eff_gb": 904.3,
        "mem_eff_pct": 79.60,
        "mem_local_gb": 635.2,
        "mem_local_pct": 55.91
    }
]

top10_idle_disk_region = [
    {
        "vnf_instance": "EDA11",
        "dc": "Nextengo Nacional",
        "disk_reserved_gb": 4750.0,
        "disk_idle_gb": 4662.8,
        "disk_idle_pct": 98.16,
        "disk_used_gb": 87.2,
        "disk_used_pct": 1.84
    },
    {
        "vnf_instance": "CUDB11",
        "dc": "Popotla",
        "disk_reserved_gb": 8540.0,
        "disk_idle_gb": 2832.8,
        "disk_idle_pct": 33.17,
        "disk_used_gb": 5707.2,
        "disk_used_pct": 66.83
    },
    {
        "vnf_instance": "EPDG01",
        "dc": "San Juan",
        "disk_reserved_gb": 1360.0,
        "disk_idle_gb": 1040.9,
        "disk_idle_pct": 76.53,
        "disk_used_gb": 319.2,
        "disk_used_pct": 23.47
    },
    {
        "vnf_instance": "PGWU01",
        "dc": "Nextengo Regional",
        "disk_reserved_gb": 1280.0,
        "disk_idle_gb": 584.1,
        "disk_idle_pct": 45.63,
        "disk_used_gb": 695.9,
        "disk_used_pct": 54.37
    },
    {
        "vnf_instance": "BGF01",
        "dc": "San Juan",
        "disk_reserved_gb": 216.0,
        "disk_idle_gb": 167.1,
        "disk_idle_pct": 77.33,
        "disk_used_gb": 49.0,
        "disk_used_pct": 22.67
    },
    {
        "vnf_instance": "PGWC01",
        "dc": "Popotla",
        "disk_reserved_gb": 560.0,
        "disk_idle_gb": 151.1,
        "disk_idle_pct": 26.99,
        "disk_used_gb": 408.9,
        "disk_used_pct": 73.01
    },
    {
        "vnf_instance": "AAA01",
        "dc": "Nextengo Nacional",
        "disk_reserved_gb": 560.0,
        "disk_idle_gb": 93.5,
        "disk_idle_pct": 16.69,
        "disk_used_gb": 466.5,
        "disk_used_pct": 83.31
    },
    {
        "vnf_instance": "PCRF02",
        "dc": "Nextengo Regional",
        "disk_reserved_gb": 200.1,
        "disk_idle_gb": 89.1,
        "disk_idle_pct": 44.52,
        "disk_used_gb": 111.0,
        "disk_used_pct": 55.48
    },
    {
        "vnf_instance": "PCRF01",
        "dc": "San Juan",
        "disk_reserved_gb": 200.1,
        "disk_idle_gb": 88.5,
        "disk_idle_pct": 44.22,
        "disk_used_gb": 111.6,
        "disk_used_pct": 55.78
    },
    {
        "vnf_instance": "IPWorks01",
        "dc": "Popotla",
        "disk_reserved_gb": 560.0,
        "disk_idle_gb": 87.0,
        "disk_idle_pct": 15.53,
        "disk_used_gb": 473.1,
        "disk_used_pct": 84.47
    }
]

geo_redundancy_region = [
    {
        "vnf": "AAA",
        "sites": ["Revolución", "Madero", "San Juan", "Nextengo"],
        "redundancy_type": "Pool"
    },
    {
        "vnf": "AFG",
        "sites": ["Revolución", "Nextengo"],
        "redundancy_type": "Pool"
    },
    {
        "vnf": "BGF",
        "sites": ["San Juan", "Nextengo"],
        "redundancy_type": "Pool"
    },
    {
        "vnf": "CSCF",
        "sites": ["San Juan", "Nextengo"],
        "redundancy_type": "Pool"
    },
    {
        "vnf": "CUDB",
        "sites": ["San Juan", "Nextengo"],
        "redundancy_type": "Pool"
    },
    {
        "vnf": "DRA",
        "sites": ["San Juan", "Nextengo"],
        "redundancy_type": "Pool"
    },
    {
        "vnf": "EDA",
        "sites": ["San Juan", "Nextengo"],
        "redundancy_type": "Activo-Pasivo"
    },
    {
        "vnf": "EPDG",
        "sites": ["Revolución", "Mayo", "Nextengo", "San Juan"],
        "redundancy_type": "Pool"
    },
    {
        "vnf": "HLR",
        "sites": ["Nextengo", "Popotla"],
        "redundancy_type": "Pool"
    },
    {
        "vnf": "HSS",
        "sites": ["Popotla", "Nextengo"],
        "redundancy_type": "Pool"
    },
    {
        "vnf": "IPWorks",
        "sites": ["San Juan", "Nextengo"],
        "redundancy_type": "Pool"
    },
    {
        "vnf": "MME",
        "sites": ["Popotla", "San Juan", "Nextengo"],
        "redundancy_type": "Pool"
    },
    {
        "vnf": "MRF",
        "sites": ["Nextengo", "San Juan"],
        "redundancy_type": "Pool"
    },
    {
        "vnf": "MTAS",
        "sites": ["San Juan", "Nextengo"],
        "redundancy_type": "Pool"
    },
    {
        "vnf": "PDRA",
        "sites": ["San Juan", "Nextengo"],
        "redundancy_type": "Pool"
    },
    {
        "vnf": "PGWC",
        "sites": ["San Juan", "Nextengo", "Popotla"],
        "redundancy_type": "Pool"
    },
    {
        "vnf": "PGWU",
        "sites": ["San Juan", "Popotla", "Nextengo"],
        "redundancy_type": "Pool"
    },
    {
        "vnf": "SBG",
        "sites": ["San Juan", "Nextengo"],
        "redundancy_type": "Pool"
    },
    {
        "vnf": "UPG",
        "sites": ["Revolución", "Nextengo"],
        "redundancy_type": "Pool"
    }
]