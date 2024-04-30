import random
import math

"""A class represnting a node in an AVL tree"""

class AVLNode(object):
	"""
	initializes AVL node in O(1)
	@type value: str
	@param value: data of your node
	"""
	def __init__(self, value, left=None, right=None, parent=None):
		self.value = value
		self.parent = parent
		if (left != None):                    #if the node has a son dont make a virtual son
			self.left = left
			self.left.parent = self
		else:
			if (value != None):    #if the node is not virtual
				self.left = AVLNode(None, parent=self)    #build a virtual son
			else:
				self.left = None
		if (right != None):
			self.right = right
			self.right.parent = self
		else:
			if (value != None):
				self.right = AVLNode(None, parent=self)
			else:
				self.right = None
				self.parent = parent
		if (value == None):           #virtual node
			self.height = -1
			self.size = 0
		else:                       #regular node
			self.height = max(self.left.height, self.right.height) + 1
			self.size = self.left.size + self.right.size + 1



	"""returns the left child
	@rtype: AVLNode
	@returns: the left child of self, None if there is no left child
	"""
	def getLeft(self):
		return  self.left


	"""returns the right child

	@rtype: AVLNode
	@returns: the right child of self, None if there is no right child
	"""
	def getRight(self):
		return self.right

	"""returns the parent 

	@rtype: AVLNode
	@returns: the parent of self, None if there is no parent
	"""
	def getParent(self):
		return self.parent

	"""return the value

	@rtype: str
	@returns: the value of self, None if the node is virtual
	"""
	def getValue(self):
		return self.value

	def getSize(self):
		return self.size
	"""returns the height

	@rtype: int
	@returns: the height of self, -1 if the node is virtual
	"""
	def getHeight(self):
		return self.height

	"""sets left child
	
	@type node: AVLNode
	@param node: a node
	"""
	def setLeft(self, node):
		self.left = node
		return None

	"""sets right child

	@type node: AVLNode
	@param node: a node
	"""
	def setRight(self, node):
		self.right = node
		return None

	"""sets parent

	@type node: AVLNode
	@param node: a node
	"""
	def setParent(self, node):
		self.parent = node
		return None

	"""sets value

	@type value: str
	@param value: data
	"""
	def setValue(self, value):
		self.value = value
		return None

	"""sets the balance factor of the node

	@type h: int
	@param h: the height
	"""
	def setHeight(self, h):
		self.height = h
		return None

	"""returns whether self is not a virtual node 

	@rtype: bool
	@returns: False if self is a virtual node, True otherwise.
	"""

	def isRealNode(self):
		return self.getHeight() != -1

	"""returns whether self is a leaf in O(1) 
	
	@rtype: bool
	@returns: True if self is a leaf, False otherwise.
	"""

	def isLeaf(self):
		return (not self.getLeft().isRealNode()) and (not self.getRight().isRealNode())

	"""sets size
	
	@type value: str
	@param size: the size
	"""
	def setSize(self, size):
		self.size = size

	"""returns the leftmost child of which the node is an ancestor
	
	@rtype: node
	@returns: leftmost child of node
	"""
	def minOfThis(node):
		while node.left.value != None:
			node = node.left
		return node

	"""returns the rightmost child of which the node is an ancestor

	@rtype: node
	@returns: rightmost child of node
	"""

	def maxOfThis(node):
		while node.right.value != None:
			node = node.right
		return node

	"""returns the Balance Factor of the node 

	@rtype: int
	@returns: Balance factor of self
	"""

	def BF(self):
		return self.getLeft().getHeight() - self.getRight().getHeight()

"""
A class implementing the ADT list, using an AVL tree.
"""

class AVLTreeList(object):

	"""
	initializes AVL TreeList in O(1)
	"""

	def __init__(self):
		self.size = 0
		self.root = None
		self.max = None
		self.min = None

	"""returns whether the list is empty

	@rtype: bool
	@returns: True if the list is empty, False otherwise
	"""
	def empty(self):
		return self.size == 0


	"""retrieves the value of the i'th item in the list in O(log n)
	
	@type i: int
	@pre: 0 <= i < self.length()
	@param i: index in the list
	@rtype: str
	@returns: the the value of the i'th item in the list
	"""
	def retrieve(self, i):
		if self.empty() or i >= self.length() or i < 0:
			return None
		node = AVLTreeList.TreeSelect(self, i+1)
		return node.value



	"""inserts val at position i in the list in O(log n)
	
	@type i: int
	@pre: 0 <= i <= self.length()
	@param i: The intended index in the list to which we insert val
	@type val: str
	@param val: the value we inserts
	@rtype: list
	@returns: the number of rebalancing operation due to AVL rebalancing
	"""
	def insert(self, i, val):
		cnt = 0
		if self.empty():
			self.root = AVLNode(val)
			self.size = 1
			self.max = self.root
			self.min= self.root
			return 0
		if (i == self.size):
			currNode = self.max
			newNode = AVLNode(val,parent=currNode)
			currNode.setRight(newNode)
			self.max = newNode
		else:
			currNode = self.TreeSelect( i + 1)
			if (i == 0):
				newNode = AVLNode(val, parent=currNode)
				self.min = newNode
				currNode.setLeft(newNode)
			elif (not currNode.getLeft().isRealNode()):
				newNode = AVLNode(val,parent=currNode)
				currNode.setLeft(newNode)
			else:
				currNode = self.predecessor(currNode)
				newNode = AVLNode(val, parent=currNode)
				currNode.setRight(newNode)
		self.size += 1
		num_of_rotations = self.AVL_insert(newNode, cnt)
		return num_of_rotations



	"""operates AVL rebalancing after insert with rotations and 
	fixes heights and sizes in O(log n)

	@type node: node
	@type cnt: int	
	@param node: The node we inserted to the tree from which we want to start rebalancing
	@param cnt: counts the number of rotations 
	@rtype: int
	@returns: the number of rebalancing operation due to AVL rebalancing
	"""

	def AVL_insert(self, node, cnt):
		y = node.getParent()
		while y != None :
			y.setSize(y.getLeft().getSize() + y.getRight().getSize() + 1)
			y.setHeight(1 + max(y.getLeft().getHeight(), y.getRight().getHeight()))
			b_factor = abs(y.BF())

			if (b_factor < 2) :
				y = y.getParent()
				continue
			else:
				b_factor = y.BF()
				if b_factor == -2:
					if y.getRight().BF() == -1:
						A = y
						B = y.getRight()
						self.leftRotation(A, B)
						self.fix_height_size_of_AB_after_insert( A, B)  #function fixes sizes,heights after rebalancing
						cnt += 1

					elif y.getRight().BF() == 1:
						A = y.getRight().getLeft()
						B = y.getRight()
						self.rightRotation(A, B)
						self.fix_height_size_of_AB_after_insert(A, B)

						self.leftRotation(A.getParent(), A)
						self.fix_height_size_of_AB_after_insert(A.getLeft(), A)
						cnt += 2

				elif b_factor == 2:
					if y.getLeft().BF() == 1:
						A = y.getLeft()
						B = y
						self.rightRotation(A, B)
						self.fix_height_size_of_AB_after_insert(A, B)
						cnt += 1

					elif y.getLeft().BF() == -1:
						A = y.getLeft()
						B = y.getLeft().getRight()
						self.leftRotation(A, B)
						self.fix_height_size_of_AB_after_insert(A, B)

						self.rightRotation(B, B.getParent())
						self.fix_height_size_of_AB_after_insert(B, B.getRight())
						cnt += 2

			y = y.getParent()
		return cnt


	"""returns the k th smallest node in the tree (by position- 1=<k<=Tree.size()-1) in O(log n)
	
	@type k : int
	@param k: the intended position of the node we want to get  (by size)
	@rtype: AVL node
	@returns : if the tree is empty returns none,else returns the k'th smallest node
	"""
	def TreeSelect(self, k):  # self here is pointer for tree
		if self.empty():
			return None
		return self.TreeSelectNode(self.root, k)

	def TreeSelectNode(self, node, k):   # Here we get a Pointer to Node, and return k'th node
		if AVLNode.getLeft(node) is None:
			r = 1
		else:
			r = node.left.size + 1
		if k == r:
			return node
		else:
			if k < r:
				return self.TreeSelectNode( node.left, k)
			else:  # k>r
				return self.TreeSelectNode( node.right, k - r)

	"""fixes the height and size of two nodes after rotating them in O(1)

	@type A : AVL node
	@type B : AVL node
	@param A: first node participating the rotation
	@param B: second node participating the rotation
	@returns : None
	"""


	def fix_height_size_of_AB_after_insert(self,A, B):
		if A.getParent() == B:       #B is A's parent
			A.setSize(A.getLeft().getSize() + A.getRight().getSize() + 1)
			B.setSize(B.getLeft().getSize() + B.getRight().getSize() + 1)
			A.setHeight(1 + max(A.getLeft().getHeight(), A.getRight().getHeight()))
			B.setHeight(1 + max(B.getLeft().getHeight(), B.getRight().getHeight()))
		else:
			B.setSize(B.getLeft().getSize() + B.getRight().getSize() + 1)
			A.setSize(A.getLeft().getSize() + A.getRight().getSize() + 1)
			B.setHeight(1 + max(B.getLeft().getHeight(), B.getRight().getHeight()))
			A.setHeight(1 + max(A.getLeft().getHeight(), A.getRight().getHeight()))


	"""deletes the i'th item in the list in O(log n)

	@type i: int
	@pre: 0 <= i < self.length()
	@param i: The intended index in the list to be deleted
	@rtype: int
	@returns: the number of rebalancing operation due to AVL rebalancing
	"""

	def Delete(self, i):
		if i >= self.length() or i < 0 :
			return -1

		NodeToDelete = AVLTreeList.TreeSelect(self, i + 1)  # here we find the i+1'th smallest node in tree
		num_of_rotations = AVLTreeList.DeleteFromTree(self, NodeToDelete)  #here we delete the node
		self.size = self.size - 1
		if self.size == 1:     #if we're left with one node
			self.min = self.root
			self.max = self.root
		if NodeToDelete == self.min:        #if we deleted the minimum
			self.min = self.successor(NodeToDelete)
		if NodeToDelete == self.max:       #if we deleted the maximum
			self.max = self.predecessor(NodeToDelete)

		return num_of_rotations

	"""deletes the node from the tree in  O(log n)

		@type node: AVL node
		@param node: The node we want to delete
		@rtype: int
		@returns: the number of rebalancing operation due to AVL rebalancing
		"""

	def DeleteFromTree(self, node):
		cnt = 0
		fixFromNode = node
		if self.size == 1:
			self.root.setSize(0)
			self.min = None
			self.max = None
			#self.size = 0
			self.root = None
			return 0

		if node.isLeaf():
			fixFromNode = node
			y = node.getParent()
			if y.getLeft() == node:
				y.setLeft(AVLNode(None, parent=y))
			else:
				y.setRight(AVLNode(None, parent=y))
			y.setSize(y.getSize() - 1)

		elif node.getLeft().getValue() != None and node.getRight().getValue() == None:
			# has 1 son, LEFT SON
			y = node.getParent()
			if y == None:
				node.getLeft().setParent(y)
				self.root = node.getLeft()
				return None
			if y.getLeft() == node:
				y.setLeft(node.getLeft())
				node.getLeft().setParent(y)
			else:
				y.setRight(node.getLeft())
				node.getLeft().setParent(y)
			fixFromNode = node          #the node from which we want to fix the tree

		elif node.getLeft().getValue() == None and node.getRight().getValue() != None:
			y = node.getParent()
			if y == None:
				node.getRight().setParent(y)
				self.root = node.getRight()
				return None
			if y.getLeft() == node:
				y.setLeft(node.getRight())
				node.getRight().setParent(y)

			else:  # y.getRight == node
				y.setRight(node.getRight())
				node.getRight().setParent(y)

			fixFromNode = node
		else:  # has 2 sons
			y = AVLTreeList.successor(self, node)
			fixFromNode = y
			node.setValue(y.getValue())
			x = y.getParent()

			# y is x's left son
			if x.getRight() == y:
				x.setRight(y.getRight())
				y.getRight().setParent(x)
			else:
				x.setLeft(y.getRight())
				y.getRight().setParent(x)

		num_of_rotations = self.AVL_delete( fixFromNode, cnt)
		return num_of_rotations

	"""operates AVL rebalancing after delete with rotations and 
	fixes heights and sizes in O(log n)

	@type node: node
	@type cnt: int
	@param node: The node we pyisically deleted from which we want to start rebalancing
	@param cnt: counts the number of rotations 
	@rtype: int
	@returns: the number of rebalancing operation due to AVL rebalancing
	"""
	def AVL_delete(self, node, cnt):
		y = node.getParent()
		while y != None :
			y.setSize(y.getLeft().getSize() + y.getRight().getSize() + 1)
			y.setHeight(1 + max(y.getLeft().getHeight(), y.getRight().getHeight()))
			b_factor = abs(y.BF())

			if (b_factor < 2) :
				y = y.getParent()
				continue

			else:
				b_factor = y.BF()
				if b_factor == -2:
					if y.getRight().BF() == -1 or y.getRight().BF() == 0 :
						A = y
						B = y.getRight()
						self.leftRotation(A, B)
						self.fix_height_size_of_AB_after_insert( A, B) #the same function we used in insert fixing
						cnt += 1

					elif y.getRight().BF() == 1:
						A = y.getRight().getLeft()
						B = y.getRight()
						self.rightRotation(A, B)
						self.fix_height_size_of_AB_after_insert(A, B)

						self.leftRotation(A.getParent(), A)
						self.fix_height_size_of_AB_after_insert(A.getLeft(), A)

						cnt += 2
				elif b_factor == 2:
					if y.getLeft().BF() == 1 or y.getLeft().BF() == 0:
						A = y.getLeft()
						B = y
						self.rightRotation(A, B)
						self.fix_height_size_of_AB_after_insert(A, B)

						cnt += 1
					elif y.getLeft().BF() == -1:
						A = y.getLeft()
						B = y.getLeft().getRight()
						self.leftRotation(A, B)
						self.fix_height_size_of_AB_after_insert(A, B)

						self.rightRotation(B, B.getParent())
						self.fix_height_size_of_AB_after_insert(B, B.getRight())
						cnt += 2
			y = y.getParent()
		return cnt

	"""left rotation of nodes A B when A is the left 
	positioned node before the left rotation - in O(1)

	@type A : AVL node
	@type B : AVL node
	@param A: left postioned node participating  left rotation
	@param B: right positioned node participating  left rotation
	@returns : None
	"""
	def leftRotation(self, A, B):
		if A.getParent() != None:
			if A.getParent().getLeft() == A:
				check = "L"
			else:
				check = "R"
		else:
			self.root = B
			B.setParent(A.getParent())
		A.setRight(B.getLeft())
		A.getRight().setParent(A)
		B.setLeft(A)
		B.setParent(A.getParent())
		if A.getParent() != None:
			if check == "L":
				B.getParent().setLeft(B)
			else:
				B.getParent().setRight(B)
		A.setParent(B)

	"""right rotation of nodes A B when A is  the left 
	positioned node before the right rotation - in  O(1)

	@type A : AVL node
	@type B : AVL node
	@param A: left postioned node participating  right rotation
	@param B: right positioned node participating  right rotation
	@returns : None
	"""

	def rightRotation(self, A, B):
		if B.getParent() != None:
			if B.getParent().getLeft() == B:
				check = "L"
			else:
				check = "R"
		else:
			self.root = A
			A.setParent(B.getParent())
		B.setLeft(A.getRight())
		B.getLeft().setParent(B)
		A.setRight(B)
		A.setParent(B.getParent())
		if B.getParent() != None:
			if check == "L":
				A.getParent().setLeft(A)
			else:
				A.getParent().setRight(A)
		B.setParent(A)


	"""returns the value of the first item in the list in O(1)

	@rtype: str
	@returns: the value of the first item, None if the list is empty
	"""
	def first(self):
		if self.empty() or self.min == None:
			return None
		return self.min.getValue()

	"""returns the value of the last item in the list in O(1)
	
	@rtype: str
	@returns: the value of the last item, None if the list is empty
	"""
	def last(self):
		if self.empty() or self.max == None:
			return None
		return self.max.getValue()

	"""returns an array representing list in O(n)

	@rtype: list
	@returns: a list of strings representing the data structure
	"""
	def listToArray(self):
		lst = []
		if self.empty():
			return lst
		self.sortValuesOfTree(self.getRoot(), lst)
		return lst

	"""Auxiliary function to listToArray
	in order walk over the tree adding the values of the nodes to the list -in O(n)
	
	@type root: AVL node
	@type lst: list
	@param root: pointer to the root of the tree we want to add to the list
	@param lst: the list we want to return in listToArray
	@returns: None
	"""
	def sortValuesOfTree(self, root, lst):
		if root.isRealNode() == False:
			return None
		self.sortValuesOfTree(root.getLeft(), lst)
		# adding left side, then root, then right side,
		# literraly like traversal walk in O(n)!
		lst.append(root.getValue())
		self.sortValuesOfTree(root.getRight(), lst)

	"""returns the size of the list 
	
	@rtype: int
	@returns: the size of the list
	"""
	def length(self):
		if self.empty():
			return 0
		return self.size

	"""sort the info values of the list - in in O(n log n)

	@rtype: list
	@returns: an AVLTreeList where the values are sorted by the info of the original list.
	"""
	def sort(self):
		if self.size == 1:
			Tr = AVLTreeList()
			Node = self.getRoot()
			Tr.insert(0, Node.getValue())
			return Tr
		if self.size == 0:
			x = AVLTreeList()
			return x
		sortedList = self.mergeSort()  # supposed to return sorted list
		newAvlSorted = AVLTreeList()
		j = 0
		for item in sortedList:  # O(n)
			newAvlSorted.insert(j, item)  # O(logn)
			j += 1
		return (newAvlSorted)

	"""Auxiliary function to sort
	sorting a list in the natural order of the values  - in O(n log n)

	@rtype: list 
	@returns: a sorted list 
	"""
	def mergeSort(self):
		myList = self.listToArray()

		def mergeSort_rec(myList):
			if len(myList) > 1:
				mid = len(myList) // 2
				left = myList[:mid]
				right = myList[mid:]

				# Recursive call on each half
				mergeSort_rec(left)
				mergeSort_rec(right)

				# Two iterators for traversing the two halves
				i = 0
				j = 0

				# Iterator for the main list
				k = 0

				while i < len(left) and j < len(right):
					if left[i] <= right[j]:
						# The value from the left half has been used
						myList[k] = left[i]
						# Move the iterator forward
						i += 1
					else:
						myList[k] = right[j]
						j += 1
					# Move to the next slot
					k += 1

				# For all the remaining values
				while i < len(left):
					myList[k] = left[i]
					i += 1
					k += 1

				while j < len(right):
					myList[k] = right[j]
					j += 1
					k += 1
				return myList

		return mergeSort_rec(myList)

	"""permute the info values of the list 

	@rtype: list
	@returns: an AVLTreeList where the values are permuted randomly by the info of the original list. ##Use Randomness
	"""
	def permutation(self):
		if self.size == 0:
			return AVLTreeList()
		if self.size == 1:
			x = self.retrieve(0)
			Tree = AVLTreeList()
			Tree.insert(0, x)
			return Tree
		if self.size == 2:
			x = self.retrieve(0)
			y = self.retrieve(1)
			rand = random.randint(0, 1)
			Tree = AVLTreeList()
			Tree.insert(abs(1 - rand), x)
			Tree.insert(rand, y)
			return Tree
		tmpArr2 = self.listToArray()
		for i in range(self.size):
			rand = random.randint(i, self.length() - 1)
			b = rand
			AVLTreeList.swap(tmpArr2, i, rand)
		n = len(tmpArr2)
		middle = n // 2
		start = 0
		end = n - 1
		Tree = AVLTreeList()
		node = AVLNode(tmpArr2[middle])
		Tree.root = node
		Tree.root.setSize(Tree.root.getSize() + 1)
		Tree.size += 1
		Tree.min = node
		Tree.max = node
		a = AVLTreeList.ArrayToAvl(tmpArr2, start, middle - 1)
		b = AVLTreeList.ArrayToAvl(tmpArr2, middle + 1, end)
		Tree.root.setLeft(a.root)
		a.root.setParent(Tree.root)
		Tree.root.setRight(b.root)
		b.root.setParent(Tree.root)
		Tree.root.setHeight(max(a.root.height, b.root.height) + 1)
		Tree.root.setSize(a.size + b.size + 1)
		Tree.min = a.min
		Tree.max = b.max
		return Tree

	"""Auxiliary function to permutation 
	creates an AVLtree from array list in O(n)
	
	@type lst: list
	@type start: int
	@type end: int
	@param start: index of the start of the list
	@param end:  index of the end of the list
	@rtype: AVLTreeList 
	@returns: an AVLTreeList representing the lst
	"""
	def ArrayToAvl(lst, start, end):
		Tree = AVLTreeList()
		middle = math.floor((start - end + 1) / 2)
		if end - start + 1 > 3:
			node = AVLNode(lst[middle])
			a = start
			b = end
			Tree.root = node
			a = AVLTreeList.ArrayToAvl(lst, a, middle - 1)
			b = AVLTreeList.ArrayToAvl(lst, middle + 1, b)
			Tree.root.setLeft(a.root)
			a.root.setParent(Tree.root)
			Tree.root.setRight(b.root)
			b.root.setParent(Tree.root)
			Tree.root.setHeight(max(a.root.height, b.root.height) + 1)
			Tree.root.setSize(a.size + b.size + 1)
			Tree.min = a.min
			Tree.max = b.max
			Tree.size = a.size + b.size + 1
			return Tree
		if end - start + 1 == 3:
			Tree.root = AVLNode(lst[start + 1])
			a = AVLNode(lst[start])
			b = AVLNode(lst[end])
			Tree.root.setLeft(a)
			Tree.root.setRight(b)
			b.setParent(Tree.root)
			a.setParent(Tree.root)
			Tree.root.setSize(3)
			Tree.root.setHeight(1)
			Tree.size = 3
			Tree.min = a
			Tree.min = b
			return Tree
		if end - start + 1 == 1:
			Tree = AVLTreeList()
			a = AVLNode(lst[start])
			Tree.root = a
			Tree.root.setSize(1)
			Tree.min = a
			Tree.max = a
			Tree.size = 1
			return Tree
		if end - start + 1 == 2:
			a = AVLNode(lst[start])
			b = AVLNode(lst[end])
			b.setLeft(a)
			a.setParent(b)
			Tree.root = b
			Tree.root.setSize(2)
			Tree.min = a
			Tree.max = b
			Tree.size = 2
			return Tree

	"""Auxiliary function to permutation 
	given  a list and two indexes, swaping the items of these indexes  -in O(1)

	@type self: list
	@type i: int
	@type j: int
	@param i: index of first item we want to swap
	@param j: index of second item we want to swap
	@returns: None
	"""

	def swap(self, i, j):
		a = self[i]
		b = self[j]
		self[i] = b
		self[j] = a


	"""concatenates lst to self

	@type lst: AVLTreeList
	@param lst: a list to be concatenated after self
	@rtype: int
	@returns: the absolute value of the difference between the height of the AVL trees joined
	"""
	def concat(self, lst):
		if lst.empty() :
			if self.empty():
				return 0
			return self.getRoot().getHeight()
		if self.empty():
			self.root = lst.getRoot()
			self.size = lst.size
			self.min = lst.min
			self.max = lst.max
			return lst.getRoot().getHeight()
		dif = abs(self.getRoot().getHeight() - lst.getRoot().getHeight())
		if lst.length() == 1:
			self.insert(self.length() , lst.getRoot().getValue())
			self.max = lst.getRoot()
			return dif
		if self.length() == 1:
			minN= self.min
			lst.insert(0 ,self.getRoot().getValue())
			self.root = lst.getRoot()
			self.size = lst.size
			self.min = minN
			self.max = lst.max
			return dif
		flag = False

		maxN = lst.max
		if self.length() <= lst.length():
			if self.length() == lst.length():
				flag = True
			nodeT2 = lst.getRoot()
			nodeT1 = self.getRoot()
			while nodeT1.getHeight() < nodeT2.getHeight():

				nodeT2 = nodeT2.getLeft()
			minTemp = lst.min
			lst.Delete(0)
			minTemp.setParent(nodeT2.getParent())
			if nodeT2.getParent() != None:
				nodeT2.getParent().setLeft(minTemp)

			minTemp.setRight(nodeT2)
			nodeT2.setParent(minTemp)
			minTemp.setLeft(nodeT1)
			nodeT1.setParent(minTemp)

			if flag == True:      # the trees were the same height
				lst.root = minTemp
			minTemp.setSize(minTemp.getLeft().getSize() + minTemp.getRight().getSize() + 1)
			minTemp.setHeight(1 + max(minTemp.getLeft().getHeight(), minTemp.getRight().getHeight()))
			lst.AVL_delete(minTemp, cnt=0)    #AVL_delete makes a balanced tree after delete (the same need for concat)
			self.root = lst.getRoot()

		elif self.length() > lst.length():
			nodeT2 = lst.getRoot()
			nodeT1 = self.getRoot()
			while nodeT1.getHeight() > nodeT2.getHeight():
				nodeT1 = nodeT1.getRight()
			maxTemp = self.max
			self.Delete(self.length()-1)
			maxTemp.setParent(nodeT1.getParent())
			if nodeT1.getParent() != None:
				nodeT1.getParent().setRight(maxTemp)
			maxTemp.setRight(nodeT2)
			nodeT2.setParent(maxTemp)
			maxTemp.setLeft(nodeT1)
			nodeT1.setParent(maxTemp)

			maxTemp.setSize(maxTemp.getLeft().getSize() + maxTemp.getRight().getSize() + 1)
			maxTemp.setHeight(1 + max(maxTemp.getLeft().getHeight(), maxTemp.getRight().getHeight()))

			self.AVL_delete(maxTemp, cnt=0) #AVL_delete makes a balanced tree after delete (the same need for concat)

		self.max = maxN
		self.size = self.getRoot().getSize()
		return dif

	"""searches for a *value* in the list

	@type val: str
	@param val: a value to be searched
	@rtype: int
	@returns: the first index that contains val, -1 if not found.
	"""
	def search(self, val):
		lst = self.listToArray()
		# we have a sorted list by index in O(n)
		j = 0
		for item in lst:
			if item == val:
				return j
			j = j + 1
		return -1


	"""returns the root of the tree representing the list

	@rtype: AVLNode
	@returns: the root, None if the list is empty
	"""
	def getRoot(self):
		if self.empty():
			return None
		return self.root

	"""finds the successor of the node
	
	@type node: AVL node
	@param node: The node we want to find it's successor
	@rtype: AVL node
	@returns: the successor of -node-
	"""
	def successor(self, node):
		t = node.getRight()
		if t.isRealNode() :
			return t.minOfThis()
		else:  # t.value == None
			p = node.getParent()
			while p != None and node == p.getRight():
				node = p
				p = p.getParent()
		return p

	"""finds the predecessor of the node

	@type node: AVL node
	@param node: The node we want to find it's predecessor
	@rtype: AVL node
	@returns: the predecessor of -node-
	"""
	def predecessor(self, node):
		t = AVLNode.getLeft(node)
		if t.isRealNode() :
			return  t.maxOfThis()

		y = node.getParent()
		while y.isRealNode() and node == y.getLeft() :
			node = y
			y = node.getParent()
		return y






