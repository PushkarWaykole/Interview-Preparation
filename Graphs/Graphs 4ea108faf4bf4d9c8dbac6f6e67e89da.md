# Graphs

[https://www.youtube.com/watch?v=09_LlHjoEiY&list=WL&index=2&t=10153s](https://www.youtube.com/watch?v=09_LlHjoEiY&list=WL&index=2&t=10153s)

[https://en.wikipedia.org/wiki/List_of_graph_theory_topics](https://en.wikipedia.org/wiki/List_of_graph_theory_topics)

[https://takeuforward.org/graph/striver-graph-series-top-graph-interview-questions/](https://takeuforward.org/graph/striver-graph-series-top-graph-interview-questions/)

![image-1712298731666.jpg374597538052628193.jpg](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/1bd2eda5-6d9a-4d2e-ba0b-1c287440c16c.png)

## DFS

```
void dfs(int at,vector<int> adj[],vi &vis){
    vis[at]=1;
    cout<<at<<"->";
    for(auto neighbor:adj[at]){
        if(!vis[neighbor]){
            dfs(neighbor,adj,vis);
        }
    }
}
```

```cpp
int n=5;
vector<int> adj[n];
vector<vector<int>> edges={{0,1},{1,3},{0,2},{2,4}};
for(auto it:edges){
    adj[it[0]].push_back(it[1]);
}
int start_node=0;
vi vis(n,0);
dfs(start_node, adj,vis);
```

Output: *0->1->3->2->4*

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled.png)

## BFS

```cpp
void bfs(int start,vector<int> adj[],vi vis){
    queue<int> q;
    q.push(start);
    while(!q.empty()){
        auto node=q.front();
        q.pop();
        cout<<node<<"->";
        for(auto neighbor:adj[node]){
            if(!vis[neighbor]){
                vis[neighbor]=1;
                q.push(neighbor);
            }
        }
    }
}

// output: 0->1->2->3->4
```

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%201.png)

```cpp
    {int n,m;
    cin>>m>>n;
    vector<vector<char>> grid(n,vector<char>(m,'.'));
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            cin>>grid[i][j];
        }
    }
    dispgrid(grid,n,m);
    
    int moves=bfs(grid,n,m);
    print(moves);
    }
    
    
    int bfs(vector<vector<char>> grid,int n,int m){
    
    int dx[4]={0,1,0,-1};
    int dy[4]={1,0,-1,0};
    vector<vector<int>> vis(n,vector<int>(m,0));
    queue<vector<int>> q;
    q.push({0,0,0});
    vis[0][0]=1;
    int ans=0;
    while(!q.empty()){
        auto node=q.front();
        q.pop();
        int x=node[0],y=node[1],moves=node[2];

        // cout<<x<<" "<<y<<" "<<grid[x][y]<<endl;
        if(grid[x][y]=='e'){
            return moves;
        }
        for(int i=0;i<4;i++){
            int nx=x+dx[i];
            int ny=y+dy[i];
            if(nx>=0 and nx<n and ny>=0 and ny<m){
                if(grid[nx][ny]=='#' or vis[nx][ny]==1) continue;
                vis[nx][ny]=1;
                q.push({nx,ny,moves+1});
            }
        }
    }
    return -1;

}
```

```cpp
// input 
**7 5
s..#...
.#...#.
.#.....
..##...
#.#e.#.

// output
9**
```

## Toposort

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%202.png)

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%203.png)

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%204.png)

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%205.png)

Using BFS

```cpp
int n=vertices;  
vector<int> adj[n];

vector<int> indegree(n,0);
for(int i=0;i<n;i++){
    for(auto it:adj[i]){
        indegree[it]++;
    }
}

queue<int> q;
vector<int> topo;
for(int i=0;i<n;i++){
    if(indegree[i]==0){
        q.push(i);

    }
}

while(!q.empty()){
    int node=q.front();
    q.pop();
    topo.push_back(node);
    for(auto it:adj[node]){
        indegree[it]--;
        if(indegree[it]==0){
            q.push(it);
        }
    }
}

topo with contain the topological sorting
```

Using DFS

```cpp
vector < int > topoSort(int N, vector < int > adj[]) {
  stack < int > st;
  vector < int > vis(N, 0);
  for (int i = 0; i < N; i++) {
    if (vis[i] == 0) {
      findTopoSort(i, vis, st, adj);
    }
  }
  vector < int > topo;
  while (!st.empty()) {
    topo.push_back(st.top());
    st.pop();
  }
  return topo;
}
void findTopoSort(int node, vector < int > & vis, stack < int > & st, vector < int > adj[]) {
    vis[node] = 1;

    for (auto it: adj[node]) {
      if (!vis[it]) {
        findTopoSort(it, vis, st, adj);
      }
    }
    st.push(node);
  }
```

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%206.png)

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%207.png)

## Dijkstra Algorithm → O(E*log(V)) (Non negative weights only)

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%208.png)

```cpp
#include<bits/stdc++.h>
using namespace std;
int main(){
	int n=5,m=6,source=1;
	vector<pair<int,int> > g[n+1]; 	// assuming 1 based indexing of graph
	// Dijkstra's algorithm begins from here
	priority_queue<pair<int,int>,vector<pair<int,int> >,greater<pair<int,int>>> pq;
	vector<int> distTo(n+1,INT_MAX);//1-indexed array for calculating shortest paths
	distTo[source] = 0;
	pq.push({0,source});	// (dist,source)
	while( !pq.empty() ){
		int dist = pq.top().first;
		int prev = pq.top().second;
		pq.pop();
		
		for(auto it:g[prev]){
			int next = it.first;
			int nextDist = it.second;
			if( distTo[next] > distTo[prev] + nextDist){
				distTo[next] = distTo[prev] + nextDist;
				pq.push({distTo[next], next});
			}
		}
	}
	cout << "The distances from source " << source << " are : \n";
	for(int i = 1 ; i<=n ; i++)	cout << distTo[i] << " ";
	cout << "\n";
	return 0;
}

```

## To find the shortest path

```cpp
vector<int> fpath;
for(int at=n;at!=-1;at=path[at]){
    fpath.push_back(at);
}
reverse(fpath.begin(),fpath.end());
for(auto it:fpath){
    cout<<it<<" ";
}
```

## Dijkstra with indexed priority queue

The lazy implementation of Dijkstra take O(Log n) to insert a new key, value paur and O(N) time to update its value.

Using indexed priority queue we can update the distance of a node in O(Log N) time

```cpp
#include <vector>

using namespace std;

class IndexedPriorityQueue {
private:
    vector<int> heap;
    vector<int> pos;

public:
    IndexedPriorityQueue() {}

    void insert(int key, int index) {
        heap.push_back(key);
        pos[index] = heap.size() - 1;
        swim(heap.size() - 1);
    }

    int top() {
        return heap[0];
    }

    pair<int,int> pop() {
        int key = heap[0];
        int index = pos[0];
        swap(heap[0], heap[heap.size() - 1]);
        pos[0] = pos[heap.size() - 1];
        heap.pop_back();
        sink(0);
        return make_pair(key,index);
    }

    void decreaseKey(int index, int newKey) {
        int i = pos[index];
        heap[i] = newKey;
        swim(i);
    }
    bool empty(){
        return heap.size();
    }

private:
    void swim(int i) {
        while (i > 0 && heap[i] < heap[(i - 1) / 2]) {
            swap(heap[i], heap[(i - 1) / 2]);
            pos[heap[i]] = i;
            pos[heap[(i - 1) / 2]] = (i - 1) / 2;
            i = (i - 1) / 2;
        }
    }

    void sink(int i) {
        while (2 * i + 1 < heap.size()) {
            int j = 2 * i + 1;
            if (j + 1 < heap.size() && heap[j + 1] < heap[j]) {
                j++;
            }
            if (heap[i] < heap[j]) {
                swap(heap[i], heap[j]);
                pos[heap[i]] = i;
                pos[heap[j]] = j;
                i = j;
            } else {
                break;
            }
        }
    }
};
```

## Bellman Ford

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%209.png)

```cpp
vector<int> bellman_ford(int V, vector<vector<int>>& edges, int S) {
		vector<int> dist(V, 1e8);
		dist[S] = 0;
		for (int i = 0; i < V - 1; i++) {
			for (auto it : edges) {
				int u = it[0];
				int v = it[1];
				int wt = it[2];
				if (dist[u] != 1e8 && dist[u] + wt < dist[v]) {
					dist[v] = dist[u] + wt;
				}
			}
		}
		// Nth relaxation to check negative cycle
		for (auto it : edges) {
			int u = it[0];
			int v = it[1];
			int wt = it[2];
			if (dist[u] != 1e8 && dist[u] + wt < dist[v]) {
				return { -1};
			}
		}

		return dist;
	}
```

## Floyd War shall  Algorithm

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%2010.png)

```cpp
void shortest_distance(vector<vector<int>>&matrix) {
		int n = matrix.size();
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (matrix[i][j] == -1) {
					matrix[i][j] = 1e9;
				}
				if (i == j) matrix[i][j] = 0;
			}
		}

		for (int k = 0; k < n; k++) {
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					matrix[i][j] = min(matrix[i][j],
					                   matrix[i][k] + matrix[k][j]);
				}
			}
		}

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (matrix[i][j] == 1e9) {
					matrix[i][j] = -1;
				}
			}
		}
	}
};
```

## Reconstruct shortest path from node s to e

```cpp
#include<bits/stdc++.h>
using namespace std;

vector<int> reconstruct_path(int start,int end,vector<vector<int>>&dp,vector<vector<int>>&next){
    vector<int> path;
    if(dp[start][end]==1e9){
        return path;
    }
    int at=start;
    while(at!=end){
        if(at==-1) return {};
        path.push_back(at);
        at=next[at][end];
    }
    if(next[at][end]==-1) return {};
    path.push_back(end);
    return path;
}
void floyd_warshal(vector<vector<double>>&matrix) {
		int n = matrix.size();

        // here -1 indicates no route between the nodes
        vector<vector<int>> next(n,vector<int>(n,0));
        vector<vector<int>> dp(n,vector<int>(n,0));
				for (int i = 0; i < n; i++) {
					for (int j = 0; j < n; j++) {
								dp[i][j]=matrix[i][j];
                if(matrix[i][j]!=1e9){
                    next[i][j]=j;
                }
				}
		}

		for (int k = 0; k < n; k++) {
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					if(dp[i][k]+dp[k][j]<dp[i][j]){
                        dp[i][j]=dp[i][k]+dp[k][j];
                        next[i][j]=next[i][k];
                    }
				}
			}
		}

        // detecting negative cycles
        for (int k = 0; k < n; k++) {
        	for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					if(dp[i][k]+dp[k][j]<dp[i][j]){
                        dp[i][j]=-1e9;
                        next[i][j]=-1;
                    }
				}
			}
		}
		
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                cout<<dp[i][j]<<" ";
            }
            cout<<endl;
        }
          
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                cout<<"shortest path from "<<i<<" to "<<j<<" is "<<dp[i][j]<<endl;
                vector<int> path=reconstruct_path(i,j,dp,next);
                if(path.size()==0){
                    cout<<"No valid path for "<<i<<" "<<j <<endl;
                }
                else{
                    cout<<"for node "<<i<<" to "<<j<<" -> ";
                    for(auto it:path){
                        cout<<it<<" ";
                    }
                    cout<<endl;
                }
            }
        }
}
	

int main(){
    int n=4;
    vector<vector<double>> matrix={ { 0, 5, 1e9, 10 },{ 1e9, 0, 3, 1e9 },{ 1e9, 1e9, 0, 1 },{ 1e9, 1e9, 1e9, 0 }};

    for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                cout<<matrix[i][j]<<" ";
            }
            cout<<endl;
        }
    floyd_warshal(matrix);
}
```

## Bridges and Articulation points O(V+E)

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%2011.png)

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%2012.png)

## For a graph

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%2013.png)

```cpp
#include<bits/stdc++.h>
using namespace std;
#define vi vector<int>
void dfs(int node,int parent,vi &vis,vi &ids,vi &low,int &timer,
vector<int> adj[]){

    vis[node]=1;
    timer++;
    ids[node]=low[node]=timer;
    for(auto negbr:adj[node]){
        if(negbr==parent){
            continue;
        }
        if(!vis[negbr]){
            dfs(negbr,node,vis,ids,low,timer,adj);
            low[node]=min(low[node],low[negbr]);
            if(ids[node]<low[negbr]){
                cout<<node<<" "<<negbr<<endl;
            }
        }

        else{
            low[node]=min(low[node],low[negbr]);
        }
    }
}

int main(){
        ios_base::sync_with_stdio(false);
    cin.tie(NULL);

#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif
    int n,m;
    cin>>n>>m;
    vector<int> adj[n];
    for(int i=0;i<m;i++){
        int u,v; // u and v are the nodes that are connected
        cin>>u>>v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vi ids(n,-1);
    vi low(n,-1);
    vi vis(n,0);
    int timer=0;
    for(int i=0;i<n;i++){
        if(!vis[i]){
            dfs(i,-1,vis,ids,low,timer,adj);
        }
    }
}
```

```cpp
// input
9 10
0 1
0 2
1 2
2 3
2 5
3 4
5 6
5 8
6 7
7 8

//output
3 4
2 3
2 5

```

## Articulation points

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%2014.png)

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%2015.png)

```cpp
#include<bits/stdc++.h>
using namespace std;
#define vi vector<int>
void dfs(int node,int parent,vi &vis,vi &ids,vi &low,int &timer,vector<int> adj[],vi &isart,int &outedges){

    vis[node]=1;
    timer++;
    ids[node]=low[node]=timer;
    if(node==parent) outedges++;
    for(auto it:adj[node]){
        if(it==parent){
            continue;
        }
        if(!vis[it]){
            dfs(it,node,vis,ids,low,timer,adj,isart,outedges);
            low[node]=min(low[node],low[it]);
            if(low[it]>ids[node]){
                cout<<node<<" "<<it<<endl;
            }
            if(ids[node]<=low[it]){
                isart[node]=1;
            }
        }

        else{
            low[node]=min(low[node],ids[it]);
        }
    }
}

int main(){
        ios_base::sync_with_stdio(false);
    cin.tie(NULL);

#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif
    int n,m;
    cin>>n>>m;
    vector<int> adj[n];
    for(int i=0;i<m;i++){
        int u,v; // u and v are the nodes that are connected
        cin>>u>>v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vi ids(n,-1);
    vi low(n,-1);
    vi vis(n,0);
    vi isart(n,0);
    int timer=0;
    int outedges=0;
    for(int i=0;i<n;i++){
        if(!vis[i]){
            outedges=0;
            dfs(i,-1,vis,ids,low,timer,adj,isart,outedges);
            isart[i]=(outedges>1);
        }
    }
    cout<<"articulation points are: ";
    for(int i=0;i<n;i++){
        if(isart[i]!=0) cout<<i<<" ";
    }
    cout<<endl;

}

//output:
3 4
2 3
2 5
articulation points are: 2 3 5 
	
```

## Strongly connected components

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%2016.png)

## Kosaraju’s Algorithm

```cpp
#include <bits/stdc++.h>
using namespace std;

using vi = vector<int>;
#define pb push_back

const int mx = 1e5 + 1;

// adj_t is the transpose of adj
vi adj[mx], adj_t[mx], S;
bool vis[mx];
int id[mx];

void dfs(int x, int pass, int num = 0) {
    vis[x] = true;
    vi &ad = (pass == 1) ? adj[x] : adj_t[x];
    for (const int &e : ad) {
        if (!vis[e]) dfs(e, pass, num);
    }

    S.pb(x);
    if (pass == 2) id[x] = num;
}

int main() {
    cin.tie(0)->sync_with_stdio(0);
    #ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif
    int n, m;
    cin >> n >> m;

    while (m--) {
        int a, b;
        cin >> a >> b;
        adj[a+1].pb(b+1);
        adj_t[b+1].pb(a+1);
    }

    for (int i = 1; i <= n; i++) {
        // first pass
        if (!vis[i]) dfs(i, 1);
    }

    memset(vis, false, sizeof vis);

    int components = 0;

    for (int i = n - 1; i >= 0; i--) {
        if (!vis[S[i]]) {
            components++;
            dfs(S[i], 2, components);
        }
    }

    cout << components << "\n";

    map<int,vi> mpp;
    for(int i=1;i<=n;i++){
        mpp[id[i]].push_back(i);
    }
    for(auto it:mpp){
        cout<<"Component: ->";
        for(auto i:it.second){
            cout<<i-1<<" ";
        }
        cout<<endl;
    }

}
```

First pass from 1 to n to fill the stack same as that in toposort using dfs

Second pass from n-1 to 1 to find the number of connected components.

Set the ind[node]=number of component

```cpp
for (int i = n - 1; i >= 0; i--) {
        if (!vis[S[i]]) {
            components++;
            dfs(S[i], 2, components);
        }
    }
```

Graph is :

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/53ce4b3e-3e3f-40d7-b909-d9675f38736c.png)

```cpp
Input:
8 13
6 0
6 2 
3 4
6 4
2 0
0 1
4 5
5 6
3 7
7 5
1 2
7 3
5 0

//output
3
Component: ->3 7 
Component: ->4 5 6 
Component: ->0 1 2 

```

## Trajan’s algorithm

```cpp
#include <bits/stdc++.h>
using namespace std;
/**
 * Description: Tarjan's, DFS once to generate
 * strongly connected components in topological order. $a,b$
 * in same component if both $a\to b$ and $b\to a$ exist.
 * Uses less memory than Kosaraju b/c doesn't store reverse edges.
 * Time: O(N+M)
 * Source: KACTL
 * https://github.com/kth-competitive-programming/kactl/blob/master/content/graph/SCC.h
 * Verification: https://cses.fi/problemset/task/1686/
 */
struct SCC {
    int N, ti = 0;
    vector<vector<int>> adj;
    vector<int> disc, comp, st, comps;
    void init(int _N) {
        N = _N;
        adj.resize(N), disc.resize(N), comp = vector<int>(N, -1);
    }
    void ae(int x, int y) { adj[x].push_back(y); }
    int dfs(int x) {
        int low = disc[x] = ++ti;
        st.push_back(x);  // disc[y] != 0 -> in stack
        for (int y : adj[x])
            if (comp[y] == -1) low = min(low, disc[y] ?: dfs(y));
        if (low == disc[x]) {  // make new SCC, pop off stack until you find x
            comps.push_back(x);
            for (int y = -1; y != x;) comp[y = st.back()] = x, st.pop_back();
        }
        return low;
    }
    void gen() {
        for (int i = 0; i < N; i++)
            if (!disc[i]) dfs(i);
        reverse(begin(comps), end(comps));
    }
};

int main() {
    #ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif
    int n, m, a, b;
    cin >> n >> m;

    SCC graph;
    graph.init(n);
    while (m--) {
        cin >> a >> b;
        graph.ae(a, b);
    }
    graph.gen();
    int ID[200000]{};
    int ids = 0;
    for (int i = 0; i < n; i++) {
        if (!ID[graph.comp[i]]) { ID[graph.comp[i]] = ++ids; }
    }
    cout << ids << '\n';
    for (int i = 0; i < n; i++) {
        cout << ID[graph.comp[i]] << " \n"[i == n - 1];
    }

    map<int,vector<int>> mpp;
    for(int i=0;i<n;i++){
        mpp[ID[graph.comp[i]]].push_back(i);
    }
    for(auto it:mpp){
        cout<<"Component: ->";
        for(auto i:it.second){
            cout<<i<<" ";
        }
        cout<<endl;
    }
}
```

```cpp
Input:
8 13
6 0
6 2 
3 4
6 4
2 0
0 1
4 5
5 6
3 7
7 5
1 2
7 3
5 0

//output
3
Component: ->3 7 
Component: ->4 5 6 
Component: ->0 1 2 

```

## Existence of Eulerian Paths and Circuits

Questions based on the concept

[https://leetcode.com/problems/reconstruct-itinerary/](https://leetcode.com/problems/reconstruct-itinerary/)

[https://leetcode.com/problems/valid-arrangement-of-pairs/description/](https://leetcode.com/problems/valid-arrangement-of-pairs/description/)

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%2017.png)

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%2018.png)

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%2019.png)

Conditions for Eulerian path and circuits

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%2020.png)

Code to find Eulerian path in directed graph in O(E)

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%2021.png)

```cpp
#include<bits/stdc++.h>
using namespace std;
#define vi vector<int>
bool isPathPossible(vector<int> adj[],vi &indegree,vi &outdegree,int n){
    int start=0,end=0;
    for(int i=0;i<n;i++){
        if(outdegree[i]-indegree[i]>1 or indegree[i]-outdegree[i]>1){
            return false;
        }
        else if(outdegree[i]-indegree[i]==1) {
            start++;
        }
        else if(indegree[i]-outdegree[i]==1){
            end++;
        }
    }
    return (end==0 and start==0) or (end==1 and start==1);
}

//--------------------------------------------------//

void dfs(int at,vi &path,vi &outdegree,vector<int> adj[]){
    while(outdegree[at]!=0){
        int nextNode=adj[at][--outdegree[at]];
        dfs(nextNode,path,outdegree,adj);
    }
    path.insert(path.begin(),at);
}

int32_t main(){

    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif

    int n,m;
    cin>>n>>m;
    vi adj[n];
    vi indegree(n,0),outdegree(n,0);
    while(m--){
        int a,b;
        cin>>a>>b;
        adj[a].push_back(b);
        indegree[b]++;
        outdegree[a]++;
    }
    vi path;
    bool possible=isPathPossible(adj,indegree,outdegree,n);
    if(possible){
        int start=0;
        for(int i=0;i<n;i++){
            if(outdegree[i] - indegree[i]==1){
                start=i;
                break;
            }
            if(outdegree[i]>1){
                start=i;
            }
        }
        dfs(start,path,outdegree,adj);
        print("The Eulerian path is:");
        disp(path);
    }
    else{
        print("No eulerian path possible");
    }

    

    return 0;
}
```

```cpp
//input
7 12
1 2
1 3
2 2
2 4
2 4
3 1
3 2
3 5
4 3
4 6
5 6
6 3

//output
The Eulerian path is:
1 3 5 6 3 2 4 3 1 2 2 4 6 

```

## Eulerian path for undirected graph

```cpp
/*
    Time Complexity  : O(N * M)
    Space Complexity : O(N + M)

    Where N is the total number of nodes and M is the total number of edges in the graph.
*/

void addEdge(int u, int v, vector<vector<int>> &adj)
{
    adj[u].push_back(v);
    adj[v].push_back(u);
}

void deleteEdge(int u, int v, vector<vector<int>> &adj)
{
    for (int i = 0; i < adj[u].size(); i++)
    {
        if (adj[u][i] == v)
        {
            adj[u][i] = -1;
            break;
        }
    }

    for (int i = 0; i < adj[v].size(); i++)
    {
        if (adj[v][i] == u)
        {
            adj[v][i] = -1;
            break;
        }
    }
}

// Function to count all reachable nodes from given node.
void dfs(int u, int &count, vector<vector<int>> &adj, vector<bool> &vis)
{
    vis[u] = 1;
    count++;

    for (int i = 0; i < adj[u].size(); i++)
    {
        int v = adj[u][i];
        if (v != -1 and vis[v] == false)
        {
            dfs(v, count, adj, vis);
        }
    }
}

// Function to check whether given edge is Non - bridge edge or not.
bool isNonbridge(int u, int v, vector<vector<int>> &adj)
{

    int countAdjacentNodes = 0;
    for (int i = 0; i < adj[u].size(); i++)
    {
        if (adj[u][i] != -1)
        {
            countAdjacentNodes++;
        }
    }
    if (countAdjacentNodes == 1)
    {
        return true;
    }

    int n = adj.size();
    vector<bool> vis(n, false);
    int withoutRemove = 0, withRemove = 0;
    dfs(u, withoutRemove, adj, vis);

    vis = vector<bool>(n, false);
    deleteEdge(u, v, adj);
    dfs(u, withRemove, adj, vis);

    addEdge(u, v, adj);

    if (withRemove >= withoutRemove)
    {
        return true;
    }
    else
    {
        return false;
    }
}

// Function to get Euler path.
void eulerPath(int u, vector<vector<int>> &adj, vector<int> &answer)
{
    for (int i = 0; i < adj[u].size(); i++)
    {
        int v = adj[u][i];
        if (v != -1 and isNonbridge(u, v, adj) == true)
        {
            answer.push_back(v);
            deleteEdge(u, v, adj);
            eulerPath(v, adj, answer);
        }
    }
}

vector<int> printEulerPath(int n, vector<vector<int>> &edgeList)
{
    vector<int> answer;
    vector<vector<int>> adj(n);

    // Build adjacency list.
    for (int i = 0; i < edgeList.size(); i++)
    {
        int u = edgeList[i][0], v = edgeList[i][1];
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    int odd = 0;

    // Count nodes with odd degree.
    for (int i = 0; i < n; i++)
    {
        if (adj[i].size() % 2 == 1)
        {
            odd++;
        }
    }

    // Check if Euler path is not possible.
    if (odd != 0 and odd != 2)
    {
        answer.push_back(-1);
        return answer;
    }

    if (odd == 0)
    {
        answer.push_back(0);
        eulerPath(0, adj, answer);
    }
    else
    {
        for (int i = 0; i < n; i++)
        {
            if (adj[i].size() % 2 == 1)
            {
                answer.push_back(i);
                eulerPath(i, adj, answer);
                break;
            }
        }
    }

    return answer;
}
```

## Minimum Spanning Tree

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%2022.png)

## Prims algorithm using priority queue

```cpp
#include <bits/stdc++.h>
using namespace std;

class Solution
{
public:
	//Function to find sum of weights of edges of the Minimum Spanning Tree.
	int spanningTree(int V, vector<vector<int>> adj[])
	{
		priority_queue<pair<int, int>,
		               vector<pair<int, int> >, greater<pair<int, int>>> pq;

		vector<int> vis(V, 0);
		// {wt, node}
		pq.push({0, 0});
		int sum = 0;
		while (!pq.empty()) {
			auto it = pq.top();
			pq.pop();
			int node = it.second;
			int wt = it.first;

			if (vis[node] == 1) continue;
			// add it to the mst
			vis[node] = 1;
			sum += wt;
			for (auto it : adj[node]) {
				int adjNode = it[0];
				int edW = it[1];
				if (!vis[adjNode]) {
					pq.push({edW, adjNode});
				}
			}
		}
		return sum;
	}
};

int main() {

	int V = 5;
	vector<vector<int>> edges = {{0, 1, 2}, {0, 2, 1}, {1, 2, 1}, {2, 3, 2}, {3, 4, 1}, {4, 2, 2}};
	vector<vector<int>> adj[V];
	for (auto it : edges) {
		vector<int> tmp(2);
		tmp[0] = it[1];
		tmp[1] = it[2];
		adj[it[0]].push_back(tmp);

		tmp[0] = it[0];
		tmp[1] = it[2];
		adj[it[1]].push_back(tmp);
	}

	Solution obj;
	int sum = obj.spanningTree(V, adj);
	cout << "The sum of all the edge weights: " << sum << endl;

	return 0;
}

```

Kruskals Algorithm

```cpp
#include <bits/stdc++.h>
using namespace std;

class DisjointSet {
    vector<int> rank, parent, size;
public:
    DisjointSet(int n) {
        rank.resize(n + 1, 0);
        parent.resize(n + 1);
        size.resize(n + 1);
        for (int i = 0; i <= n; i++) {
            parent[i] = i;
            size[i] = 1;
        }
    }

    int findUPar(int node) {
        if (node == parent[node])
            return node;
        return parent[node] = findUPar(parent[node]);
    }

    void unionByRank(int u, int v) {
        int ulp_u = findUPar(u);
        int ulp_v = findUPar(v);
        if (ulp_u == ulp_v) return;
        if (rank[ulp_u] < rank[ulp_v]) {
            parent[ulp_u] = ulp_v;
        }
        else if (rank[ulp_v] < rank[ulp_u]) {
            parent[ulp_v] = ulp_u;
        }
        else {
            parent[ulp_v] = ulp_u;
            rank[ulp_u]++;
        }
    }

    void unionBySize(int u, int v) {
        int ulp_u = findUPar(u);
        int ulp_v = findUPar(v);
        if (ulp_u == ulp_v) return;
        if (size[ulp_u] < size[ulp_v]) {
            parent[ulp_u] = ulp_v;
            size[ulp_v] += size[ulp_u];
        }
        else {
            parent[ulp_v] = ulp_u;
            size[ulp_u] += size[ulp_v];
        }
    }
};
class Solution
{
public:
    //Function to find sum of weights of edges of the Minimum Spanning Tree.
    int spanningTree(int V, vector<vector<int>> adj[])
    {
        // 1 - 2 wt = 5
        /// 1 - > (2, 5)
        // 2 -> (1, 5)

        // 5, 1, 2
        // 5, 2, 1
        vector<pair<int, pair<int, int>>> edges;
        for (int i = 0; i < V; i++) {
            for (auto it : adj[i]) {
                int adjNode = it[0];
                int wt = it[1];
                int node = i;

                edges.push_back({wt, {node, adjNode}});
            }
        }
        DisjointSet ds(V);
        sort(edges.begin(), edges.end());
        int mstWt = 0;
        for (auto it : edges) {
            int wt = it.first;
            int u = it.second.first;
            int v = it.second.second;

            if (ds.findUPar(u) != ds.findUPar(v)) {
                mstWt += wt;
                ds.unionBySize(u, v);
            }
        }

        return mstWt;
    }
};

int main() {

    int V = 5;
    vector<vector<int>> edges = {{0, 1, 2}, {0, 2, 1}, {1, 2, 1}, {2, 3, 2}, {3, 4, 1}, {4, 2, 2}};
    vector<vector<int>> adj[V];
    for (auto it : edges) {
        vector<int> tmp(2);
        tmp[0] = it[1];
        tmp[1] = it[2];
        adj[it[0]].push_back(tmp);

        tmp[0] = it[0];
        tmp[1] = it[2];
        adj[it[1]].push_back(tmp);
    }

    Solution obj;
    int mstWt = obj.spanningTree(V, adj);
    cout << "The sum of all the edge weights: " << mstWt << endl;
    return 0;
}

```

## Max Flow Ford-Fulkerson method

Time complexity using BFS is: O(fE)

f being the maximum flow and E being the edges

Edmonds-Karps ⇒ uses BFS to find augmenting paths → O(E^2V)

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%2023.png)

Each edge has a maximum flow limit

The goal of the algorithm is to find the maximum flow possible at the **SINK** node

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%2024.png)

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%2025.png)

The backward edges are called residual edges

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%2026.png)

Example

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%2027.png)

Final output should be

![Untitled](Graphs%204ea108faf4bf4d9c8dbac6f6e67e89da/Untitled%2028.png)

```cpp
class Solution
{
public:
#define vi vector<int>

    bool bfs(vector<vi> &graph,int start,int end,vi &parent,int V){
        vi vis(V,0);
        queue<int> q;
        q.push(start);
        vis[start]=1;
        parent[start]=-1;
        while(!q.empty()){
            int u=q.front();
            q.pop();
            for(int v=0;v<V;v++){
                if(!vis[v] and graph[u][v]>0){
                    parent[v]=u;
                    if(v==end){
                        return true;
                    }
                    q.push(v);
                    vis[v]=1;
                }
            }
        }
        return false;
    }

    int fordFulkerson(vector<vector<int>>&graph,int start,int end,int V){
        int u,v;
        // vector<vector<int>> rgraph=graph;
        vi parent(V);
        int maxFlow=0;
        while(bfs(graph,start,end,parent,V)){
            int min_path_flow=INT_MAX;
            for(v=end;v!=start;v=parent[v]){
                u=parent[v];
                min_path_flow=min(min_path_flow,graph[u][v]);
            }
            for(v=end;v!=start;v=parent[v]){
                u=parent[v];
                graph[u][v]-=min_path_flow;
                graph[v][u]+=min_path_flow;
            }
            maxFlow+=min_path_flow;
        }
        return maxFlow;
    }
    
    int findMaxFlow(int N,int M,vector<vector<int>> Edges)
    {
        // code here
        // Edge of type [from,to,weight]
        vector<vector<int>> graph(N,vector<int>(N,0));
        for(auto it:Edges){
            int a=--it[0],b=--it[1];
            graph[a][b]+=it[2];
            graph[b][a]+=it[2];
        }
        int maxFlow=fordFulkerson(graph,0,N-1,N);
        return maxFlow;
    }
};
```

**Time Complexity : O(|V| * E^2)** ,where E is the number of edges and V is the number of vertices.

## Undirected Bipartite matching

Bipartite graph

```cpp
class Solution {
public:
    bool isBipartite(vector<vector<int>>& graph) {
        int n = graph.size();
        vector<int> color(n); // 0: uncolored; 1: color A; -1: color B

        queue<int> q; // queue, resusable for BFS

        for (int i = 0; i < n; i++) {
            if (color[i])
                continue; // skip already colored nodes

            // BFS with seed node i to color neighbors with opposite color
            color[i] = 1; // color seed i to be A (doesn't matter A or B)
            q.push(i);
            while(!q.empty()) {
                int cur = q.front();
                for (int neighbor : graph[cur]) {
                    if (!color[neighbor]) // if uncolored, color with opposite
                                          // color
                    {
                        color[neighbor] = -color[cur];
                        q.push(neighbor);
                    }

                    else if (color[neighbor] == color[cur])
                        return false; // if already colored with same color,
                                      // can't be bipartite!
                }
                q.pop();
            }
        }

        return true;
    }
};
```