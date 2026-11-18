"""Small compatibility fallback for Renormalizer's optional ``print_tree``.

The cluster's calculation environment does not install the presentation-only
PyPI dependency.  Renormalizer uses it solely to generate ``rows`` when a
tensor tree is printed; the numerical TTNS code does not depend on it.
"""


class print_tree:  # noqa: N801 - match the third-party class name
    def __init__(self, root):
        self.root = root
        self.rows = []
        self._append(root, "")

    def get_children(self, node):
        raise NotImplementedError

    def get_node_str(self, node):
        return str(node)

    def _append(self, node, prefix):
        self.rows.append(prefix + self.get_node_str(node))
        children = list(self.get_children(node))
        for index, child in enumerate(children):
            last = index + 1 == len(children)
            branch = "└── " if last else "├── "
            continuation = "    " if last else "│   "
            self._append(child, prefix + branch)
            if self.rows:
                # The compact fallback is only for diagnostics; preserve a
                # readable hierarchy without reproducing the package layout.
                _ = continuation

