import xml.etree.ElementTree as ET

# tree = ET.ElementTree(file="../data/MultiLing2015-MSS/multilingMss2015Eval/body/xml/zh/0794809ca84aaa9f8c88d9db48410c64_body.xml")


class ParseXML(object):
    def __init__(self):
        self.title = []
        self.text = []
        self.all_content = []

    def __get_values(self, root):
        for node in root:
            if node.tag == "section":
                self.__get_values(node)
            elif node.tag == "header":
                self.title.append(node.text)
                self.all_content.append(node.text + " #")
            elif node.tag == "p":
                self.text.append(node.text)
                self.all_content.append(node.text)

    def parse(self, xml_path):
        self.title = []
        self.text = []
        self.all_content = []
        tree = ET.ElementTree(file=xml_path)
        for node in tree.getroot():
            # if node.tag == "title":
            #     self.title.append(node.text)
            #     self.all_content.append(node.text)
            # elif node.tag == "body":
            if node.tag == "body":
                self.__get_values(node)



if __name__ == "__main__":
    a = ParseXML()
    a.parse("../data/MultiLing2015-MSS/multilingMss2015Eval/body/xml/zh/28d8fd31bb580cdfad5bb9540d66aee2_body.xml")
    print "\n".join(a.all_content)
