import xml.etree.ElementTree as ET


def scale_urdf(input_file, output_file, scale_factor):
    # Parse the URDF file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Scale the mesh files and joint limits
    for elem in root.iter():
        if elem.tag in ["mesh", "box", "cylinder", "sphere"]:
            if "scale" in elem.attrib:
                original_scale = [float(x) for x in elem.attrib["scale"].split()]
                new_scale = [
                    str(x * scale_factor[i]) for i, x in enumerate(original_scale)
                ]
                elem.attrib["scale"] = " ".join(new_scale)
            else:
                elem.attrib["scale"] = (
                    f"{scale_factor[0]} {scale_factor[1]} {scale_factor[2]}"
                )

        elif elem.tag == "origin":
            if "xyz" in elem.attrib:
                xyz = [float(x) for x in elem.attrib["xyz"].split()]
                new_xyz = [str(x * scale_factor[i]) for i, x in enumerate(xyz)]
                elem.attrib["xyz"] = " ".join(new_xyz)

        elif elem.tag == "limit":
            for i, x in enumerate(scale_factor):
                if "lower" in elem.attrib:
                    elem.attrib["lower"] = str(
                        float(elem.attrib["lower"]) * scale_factor[i]
                    )
                if "upper" in elem.attrib:
                    elem.attrib["upper"] = str(
                        float(elem.attrib["upper"]) * scale_factor[i]
                    )

    # Write the scaled URDF to a new file
    tree.write(output_file, encoding="utf-8", xml_declaration=True)


# Usage
input_file = (
    "/home/haoyang/project/haoyang/openvlp/src/openvlp/assets/env/laptop/mobility.urdf"
)
output_file = "/home/haoyang/project/haoyang/openvlp/src/openvlp/assets/env/laptop/scaled_output.urdf"
scale_factor = [0.2, 0.2, 0.2]

scale_urdf(input_file, output_file, scale_factor)
print(f"Scaled URDF has been saved to {output_file}")
