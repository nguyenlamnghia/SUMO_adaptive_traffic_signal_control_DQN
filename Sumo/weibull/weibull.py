import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Tham số mô phỏng
total_time = 7200  # Tổng thời gian mô phỏng (giây)
shape_param = 2  # Tham số hình dạng của Weibull (k = 2)
scale_param = total_time / 2  # Tham số tỷ lệ, đỉnh lưu lượng ở giữa khoảng thời gian
num_bins = 100  # Số khoảng thời gian

# Cấu hình các luồng từ file datn.flows.xml
flows = [
    {"id": "flow1", "type": "car", "from": "HQC_J4_J3", "vehsPerHour": 100, "total_vehicles": 200},
    {"id": "flow2", "type": "motorbike", "from": "HQC_J4_J3", "vehsPerHour": 525, "total_vehicles": 1050},
    {"id": "flow3", "type": "bus", "from": "HQC_J4_J3", "vehsPerHour": 5, "total_vehicles": 10},
    {"id": "flow4", "type": "truck", "from": "HQC_J4_J3", "vehsPerHour": 1, "total_vehicles": 2},
    {"id": "flow5", "type": "car", "from": "HQC_J2_J3", "vehsPerHour": 100, "total_vehicles": 200},
    {"id": "flow6", "type": "motorbike", "from": "HQC_J2_J3", "vehsPerHour": 612, "total_vehicles": 1224},
    {"id": "flow7", "type": "bus", "from": "HQC_J2_J3", "vehsPerHour": 6, "total_vehicles": 12},
    {"id": "flow8", "type": "truck", "from": "HQC_J2_J3", "vehsPerHour": 2, "total_vehicles": 4},
    {"id": "flow9", "type": "car", "from": "PVB_J6_J3", "vehsPerHour": 500, "total_vehicles": 1000},
    {"id": "flow10", "type": "motorbike", "from": "PVB_J6_J3", "vehsPerHour": 2500, "total_vehicles": 7599},
    {"id": "flow11", "type": "bus", "from": "PVB_J6_J3", "vehsPerHour": 4, "total_vehicles": 8},
    {"id": "flow12", "type": "truck", "from": "PVB_J6_J3", "vehsPerHour": 6, "total_vehicles": 12},
    {"id": "flow13", "type": "car", "from": "PVB_J0_J3", "vehsPerHour": 455, "total_vehicles": 910},
    {"id": "flow14", "type": "motorbike", "from": "PVB_J0_J3", "vehsPerHour": 2235, "total_vehicles": 4470},
    {"id": "flow15", "type": "bus", "from": "PVB_J0_J3", "vehsPerHour": 3, "total_vehicles": 6},
    {"id": "flow16", "type": "truck", "from": "PVB_J0_J3", "vehsPerHour": 5, "total_vehicles": 10},
]

# Danh sách để lưu tất cả các flow trước khi sắp xếp
all_flows = []
bin_edges = np.linspace(0, total_time, num_bins + 1)

# Sinh thời gian xuất hiện cho từng luồng
np.random.seed(42)  # Đặt seed để tái lập kết quả

for flow in flows:
    total_vehicles = flow["total_vehicles"]
    if total_vehicles == 0:
        continue  # Bỏ qua nếu không có xe
    
    # Sinh thời gian xuất hiện theo phân phối Weibull
    vehicle_times = np.random.weibull(shape_param, total_vehicles/2) * scale_param
    vehicle_times = np.sort(vehicle_times)  # Sắp xếp theo thời gian
    
    # Chia thành các bins
    bin_counts, _ = np.histogram(vehicle_times, bins=bin_edges)
    
    # Tạo thông tin flow cho mỗi bin có xe
    for i, count in enumerate(bin_counts):
        if count == 0:
            continue
        begin = bin_edges[i]
        end = bin_edges[i + 1]
        all_flows.append({
            "id": f"{flow['id']}_bin{i}",
            "type": flow["type"],
            "from": flow["from"],
            "begin": begin,
            "end": end,
            "number": count
        })

# Sắp xếp các flow theo thời gian begin
all_flows.sort(key=lambda x: x["begin"])

# Tạo file XML
root = ET.Element("routes")

for flow_info in all_flows:
    flow_elem = ET.SubElement(root, "flow")
    flow_elem.set("id", flow_info["id"])
    flow_elem.set("type", flow_info["type"])
    flow_elem.set("from", flow_info["from"])
    flow_elem.set("begin", str(flow_info["begin"]))
    flow_elem.set("end", str(flow_info["end"]))
    flow_elem.set("number", str(flow_info["number"]))

# Lưu file XML
xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
with open("weibull_flows.xml", "w") as f:
    f.write(xml_str)

print("File weibull_flows.xml đã được tạo thành công!")