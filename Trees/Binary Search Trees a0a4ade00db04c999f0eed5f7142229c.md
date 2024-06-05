# Binary Search Trees

## Every node has the following property

# Left node value is less than the node value ,and Right node value is greater than the node value

![Untitled](Binary%20Search%20Trees%20a0a4ade00db04c999f0eed5f7142229c/Untitled.png)

## Duplicates are not allowed Generally

## Almost the height of BST is Log N

## Search in BST

```cpp

class Solution {
public:
    TreeNode* searchBST(TreeNode* root, int val) {
        
        while(root!=NULL && root->val!=val){
            if(root->val<val){
                root=root->right;
            }
            else{
                root=root->left;
            }
        }
        return root;
    }
};
```

## Ceil in BST

[Floor and Ceil from a BST - GeeksforGeeks](https://www.geeksforgeeks.org/floor-and-ceil-from-a-bst/)

```cpp
int findCeil(BinaryTreeNode<int> *root, int x){
    // Write your code here.
    int ceil=-1;
    
    while(root){
        if(root->data==x){
            ceil=root->data;
            return ceil;
        }
        else if(x > root->data){
            root=root->right;
        }
        else{
            ceil=root->data;
            root=root->left;
        }
    }
    return ceil;
}
```

## Floor in BST

![Untitled](Binary%20Search%20Trees%20a0a4ade00db04c999f0eed5f7142229c/Untitled%201.png)

```cpp
/************************************************************

    Following is the TreeNode class structure

    template <typename T>
    class TreeNode {
       public:
        T val;
        TreeNode<T> *left;
        TreeNode<T> *right;

        TreeNode(T val) {
            this->val = val;
            left = NULL;
            right = NULL;
        }
    };

************************************************************/

int floorInBST(TreeNode<int> * root, int key)
{
    // Write your code here.
    int floor=-1;
    while(root){
        if(root->val==key){
            floor=root->val;
            return floor;
        }
        
        if(key>root->val){
            floor=root->val;
            root=root->right;
        }
        else{
            root=root->left;
        }
    }
    
    return floor;
    
}
```

# Insertion in BST

![Untitled](Binary%20Search%20Trees%20a0a4ade00db04c999f0eed5f7142229c/Untitled%202.png)

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    TreeNode* insertIntoBST(TreeNode* root, int val) {
        if(root==NULL){
            TreeNode* newNode=new TreeNode();
            newNode->val=val;
            newNode->left=newNode->right=NULL;
            return newNode;
        }
        else if(val<root->val){
            root->left=insertIntoBST(root->left,val);
        }
        else if(val>root->val){
            root->right=insertIntoBST(root->right,val);
        }
        return root;
    }
};
```

## Deletion in BST

![Untitled](Binary%20Search%20Trees%20a0a4ade00db04c999f0eed5f7142229c/Untitled%203.png)

## Kth smallest element in BST

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    
    void inorder(TreeNode* root,int k,int &count,int &ans){
        if(!root) return;
        
        inorder(root->left,k,count,ans);
        count++;
        if(count==k){
            ans=root->val;
            return;
        }
        inorder(root->right,k,count,ans);
    }
    
    int kthSmallest(TreeNode* root, int k) {
        int ans=0,count=0;
        inorder(root,k,count,ans);
        
        return ans;
    }
};
```

## Validate BST

```cpp

#include<bits/stdc++.h>
class Solution {
public:
    
    bool helper(TreeNode* root,long min,long max){
        if(!root) return true;
        
        if(root->val<=min || root->val>=max){
            return false;
        }
        
        return helper(root->left,min,root->val) &&
            helper(root->right,root->val,max);
    }
    
    
    bool isValidBST(TreeNode* root) {
        if(!root) return true;
        return helper(root,LONG_MIN,LONG_MAX);
    }
};
```

## LCA in BST

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */

class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(!root) return NULL;
    
        int data=root->val;
        if(data<p->val && data<q->val){
            return lowestCommonAncestor(root->right,p,q);
        }
        if(data>p->val && data>q->val){
            return lowestCommonAncestor(root->left,p,q);
        }
        
        return root;
    }
};
```

## Construct BST from preorder

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */

#include<bits/stdc++.h>
class Solution {
public:
    
    TreeNode* helper(vector<int> &preorder,int &i,int bound){
        if(i==preorder.size() || preorder[i]>bound){
            return NULL;
        }
        
        TreeNode* root=new TreeNode(preorder[i++]);
        root->left=helper(preorder,i,root->val);
        root->right=helper(preorder,i,bound);
        return root;
    }
    
    
    TreeNode* bstFromPreorder(vector<int>& preorder) {
       int i=0;
        return helper(preorder,i,INT_MAX);
    }
};
```

## Construct BST from inorder

```cpp
/**********************************************************

	Following is the Binary Tree Node class structure

	template <typename T>
	class BinaryTreeNode {
    	public : 
    	T data;
    	BinaryTreeNode<T> *left;
    	BinaryTreeNode<T> *right;

    	BinaryTreeNode(T data) {
        	this -> data = data;
        	left = NULL;
        	right = NULL;
    	}
	};

***********************************************************/
BinaryTreeNode<int>* helper(int *input,int low,int high){
    
    if(low>high){
        return NULL;
    }
    
        int mid=(low+high)/2;
    	BinaryTreeNode<int>* root = new BinaryTreeNode<int>(input[mid]);

        root->left=helper(input,low,mid-1);
        root->right=helper(input,mid+1,high);

        return root;
    
    
    
}

BinaryTreeNode<int>* constructTree(int *input, int n) {
	// Write your code here
    if(n==0) return NULL;
    
    return helper(input,0,n-1);
    
    
}
```

## Inorder successor in BST

a val value which is just grater than than given val

Approach 1

Find complete inorder and return the element present after the given val

Approach 2

Perform inorder traversal and the moment you get a value which is greater then the given value return that node

Time complexity O(N)

```cpp
/*************************************************************

    Following is the Binary Tree node structure

    template <typename T>

    class BinaryTreeNode
    {
    public :
        T data;
        BinaryTreeNode<T> *left;
        BinaryTreeNode<T> *right;

        BinaryTreeNode(T data) {
            this -> data = data;
            left = NULL;
            right = NULL;
        }

        ~BinaryTreeNode() {
            if (left)
            {
                delete left;
            }
            if (right)
            {
                delete right;
            }
        }
    };

*************************************************************/

void inorder(BinaryTreeNode<int> *root, vector<int> &inorderArray)
{
    if (root == NULL)
    {
        return;
    }

    inorder(root->left, inorderArray);

    inorderArray.push_back(root->data);

    inorder(root->right, inorderArray);
}

pair<int, int> predecessorSuccessor(BinaryTreeNode<int> *root, int key)
{
    // To store the inorder traversal of the BST.
    vector<int> inorderArray;

    inorder(root, inorderArray);

    int predecessor = -1, successor = -1;

    for (int i = 0; i < inorderArray.size(); i++)
    {
        if (inorderArray[i] == key)
        {
            // If predecessor exist.
            if (i - 1 >= 0)
            {
                predecessor = inorderArray[i - 1];
            }

            // If successor exist.
            if (i + 1 < inorderArray.size())
            {
                successor = inorderArray[i + 1];
            }
            break;
        }
    }

    return {predecessor, successor};
}
```

Approach 3

maintain a variable successor and every time we encounter a node with val greater than the given val we update the successor with it.

**Time complexity O(Height of tree)**

```cpp
class Solution{
  public:
    // returns the inorder successor of the Node x in BST (rooted at 'root')
    Node * inOrderSuccessor(Node *root, Node *x)
    {
        //Your code here
        Node* successor=NULL;
        while(root!=NULL){
            
            
            if(x->data >= root->data){
                root=root->right;
            }
            else{
                successor=root;
                root=root->left;
            }
        }
        return successor;
    }
};
```

## Bst Iterator

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class BSTIterator {
public:
    int pointer;
    int capacity;
    vector<int> nodes;
    void inorder(TreeNode* root,vector<int> &nodes){
        if(!root) return;
        
        inorder(root->left,nodes);
        nodes.push_back(root->val);
        inorder(root->right,nodes);
    }
    BSTIterator(TreeNode* root) {
        pointer=-1;   
        inorder(root,nodes);
        capacity=nodes.size();
    }
    
    int next() {
        pointer+=1;
        return nodes[pointer];
    }
    
    bool hasNext() {
        return (pointer+1)<capacity;
    }
};

/**
 * Your BSTIterator object will be instantiated and called as such:
 * BSTIterator* obj = new BSTIterator(root);
 * int param_1 = obj->next();
 * bool param_2 = obj->hasNext();
 */
```

Strivers solution

![Untitled](Binary%20Search%20Trees%20a0a4ade00db04c999f0eed5f7142229c/Untitled%204.png)

## Two SUM in BST

Approach 1

Find the inorder and then find two sum using 2 pointers

Time complexity: O(N) + O(N)

Space complecity: O(N)

Approach 2

Using stack like the previous question and find next and before

```cpp
class BSTIterator {
    stack<TreeNode *> myStack;
    bool reverse = true; 
public:
    BSTIterator(TreeNode *root, bool isReverse) {
        reverse = isReverse; 
        pushAll(root);
    }

    /** @return whether we have a next smallest number */
    bool hasNext() {
        return !myStack.empty();
    }

    /** @return the next smallest number */
    int next() {
        TreeNode *tmpNode = myStack.top();
        myStack.pop();
        if(!reverse) pushAll(tmpNode->right);
        else pushAll(tmpNode->left);
        return tmpNode->val;
    }

private:
    void pushAll(TreeNode *node) {
        for(;node != NULL; ) {
             myStack.push(node);
             if(reverse == true) {
                 node = node->right; 
             } else {
                 node = node->left; 
             }
        }
    }
};
class Solution {
public:
    bool findTarget(TreeNode* root, int k) {
        if(!root) return false; 
        BSTIterator l(root, false); 
        BSTIterator r(root, true); 
        
        int i = l.next(); 
        int j = r.next(); 
        while(i<j) {
            if(i + j == k) return true; 
            else if(i + j < k) i = l.next(); 
            else j = r.next(); 
        }
        return false; 
    }
};
```

## Recover BST

2 nodes are swapped from a bst

Generate the correct BST

Brute force

Find inorder

Run a pointer across inorder and update the value of node if not matched

Time complexity: 2N + NlogN

Space Complexity: O(N)

Optimized Approach

![Untitled](Binary%20Search%20Trees%20a0a4ade00db04c999f0eed5f7142229c/Untitled%205.png)

Time Complexity: O(N)

Space complexity: O(1)

## Maximum size BST in binary tree

```cpp
#include <bits/stdc++.h>
using namespace std;

/*
Given a Binary tree, find the largest BST subtree. That is, you need to find the BST with maximum height in the given binary tree. You have to return the height of largest BST.
*/

template <typename T>
class BinaryTreeNode
{
public:
    T data;
    BinaryTreeNode<T> *left;
    BinaryTreeNode<T> *right;
    BinaryTreeNode(T data)
    {
        this->data = data;
        left = NULL;
        right = NULL;
    }
};

bool isBst(BinaryTreeNode<int> *root, int min, int max)
{
    if (!root)
        return true;

    if (root->data < min || root->data > max)
    {
        return false;
    }

    bool left = isBst(root->left, min, root->data);
    bool right = isBst(root->right, root->data, max);
    return left && right;
}

int height(BinaryTreeNode<int> *root)
{
    if (!root)
        return 0;

    int left = height(root->left);
    int right = height(root->right);

    return 1 + max(left, right);
}
int largestBSTSubtree(BinaryTreeNode<int> *root)
{
    // Write your code here
    if (isBst(root, INT_MIN, INT_MAX))
    {
        return height(root);
    }
    else
    {
        int left = largestBSTSubtree(root->left);
        int right = largestBSTSubtree(root->right);
        return max(left, right);
    }
}
```

![Untitled](Binary%20Search%20Trees%20a0a4ade00db04c999f0eed5f7142229c/Untitled%206.png)