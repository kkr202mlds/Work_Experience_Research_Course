### Trees are abstract data structures that are non-linear.
- there are many types of trees
  - binary and
  - specifically binary search trees(BST).
- trees are non - linear data structures.
- have two edges pointing to two different nodes
- If it's just a tree you're going to have more than two edges if you like.
- And these nodes in the middle here these all of these are called child nodes.
- some of these nodes have two edges and then this one right here has three edges
- for trees they cannot have cycles.
- node cannot have multiple parents.
- A tree can only have one root.
- Can't edge that points to itself.
- Height of a tree : number of levels there are in a tree leading to the leaf node
- tree structure,
  - complexity - O(logn)
### Linked List, Stack and Quenue are linear data structures.
- Having one pointer pointing to the next node 
- Leaf nodes : And the nodes at the bottom here which don't have any children.
- Head(stack header), current node and Tail with None.
### Uses of tree structures
- They're everywhere in computing
- Think of organizational charts, those are all trees (can a boss be the boss of themselves? can an employee be the boss's boss? can a team member be the project manager's manager?)
- The unix file system (the command line) is a tree structure, starting with the root directory, aptly named 'root'
- Decision trees are very popular in machine learning, a lot of algorithms are trying to 'learn' what decisions to take at every level of the tree to arrive at the right outcome
### Binary Search Tree
- Each node can have at most two children.
- left of a node has to be smaller in value than that root node and each node on the right of node is greater than that root node.
- Binary Search Trees are powerful due to these two rules
  - Search - O(h)
  - Insert - O(h)
  - Delete - O(h)
  - where h = height of the tree
- Format to remember
  - In-order   - left->root->right
  - Pre-order  - root->left->right
  - Post-order - left->right->root
