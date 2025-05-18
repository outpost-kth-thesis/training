"""
Re-enforcement learning and fine-tuning using Lora and REINFORCE
"""

from tree_sitter import Language, Parser
import tree_sitter_javascript as ts_js
from zss import simple_distance, Node


JS_LANG = Language(ts_js.language())
parser = Parser(JS_LANG)

js_code = bytes("""
function findGCD(a, b) {
    while (b !== 0) {
        const temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

const num1 = 48;
const num2 = 18;

const gcd = findGCD(num1, num2);
console.log(`The GCD of ${num1} and ${num2} is: ${gcd}`);
""", "utf8")

minified_js = bytes("""
function isPalindrome(str) {
    const cleaned = str.replace(/[^a-zA-Z0-9]/g, '').toLowerCase();
    const reversed = cleaned.split('').reverse().join('');
    return cleaned === reversed;
}

const input = "A man, a plan, a canal: Panama";
const result = isPalindrome(input);
console.log(`Is the input a palindrome? ${result}`);

""", "utf8")

def parse_code(code):
    tree = parser.parse(code)
    return tree.root_node

class ZssNode(Node):
    def __init__(self, ts_node):
        self.label = self._normalize(ts_node)
        super().__init__(self.label)
        self.children = [
            ZssNode(ts_node.child(i)) for i in range(ts_node.child_count)
        ]

    def get_children(self):
        return self.children
    

    def _normalize(self, node):
        if node.type == "identifier":
            return "IDENTIFIER"
        elif node.type in {"string", "number", "true", "false", "null"}:
            return "LITERAL"
        else:
            return node.type


tree1 = ZssNode(parse_code(js_code))
tree2 = ZssNode(parse_code(minified_js))

distance = simple_distance(tree1, tree2)
print("Tree edit distance:", distance)
