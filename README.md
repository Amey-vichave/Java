# Java

✅ 1. Sorting Algorithms
public class SortingAlgorithms {
 
   // Bubble Sort: Repeatedly swaps adjacent elements if they are in the wrong order
   public static void bubbleSort(int[] arr) {
       for (int i = 0; i < arr.length - 1; i++) { // Outer loop for passes
           for (int j = 0; j < arr.length - i - 1; j++) { // Inner loop for comparisons
               if (arr[j] > arr[j + 1]) { // Swap if left element is greater
                   int temp = arr[j];
                   arr[j] = arr[j + 1];
                   arr[j + 1] = temp;
               }
           }
       }
   }
 
   // Selection Sort: Selects minimum element and places it at correct position
   public static void selectionSort(int[] arr) {
       for (int i = 0; i < arr.length - 1; i++) {
           int minIdx = i; // Assume current as minimum
           for (int j = i + 1; j < arr.length; j++) {
               if (arr[j] < arr[minIdx]) { // Update minimum index
                   minIdx = j;
               }
           }
           // Swap min element with first element
           int temp = arr[minIdx];
           arr[minIdx] = arr[i];
           arr[i] = temp;
       }
   }
 
   // Insertion Sort: Builds sorted array one element at a time
   public static void insertionSort(int[] arr) {
       for (int i = 1; i < arr.length; i++) {
           int key = arr[i]; // Pick next element
           int j = i - 1;
           // Move elements greater than key to one position ahead
           while (j >= 0 && arr[j] > key) {
               arr[j + 1] = arr[j];
               j--;
           }
           arr[j + 1] = key; // Place key at correct position
       }
   }
 
   // Utility function to print array
   public static void printArray(int[] arr) {
       for (int num : arr) {
           System.out.print(num + " ");
       }
       System.out.println();
   }
 
   // Main function to run the sorts
   public static void main(String[] args) {
       int[] array = {64, 25, 12, 22, 11};
 
       System.out.println("Original Array:");
       printArray(array);
 
       System.out.println("Bubble Sort:");
       int[] bubble = array.clone();
       bubbleSort(bubble);
       printArray(bubble);
 
       System.out.println("Selection Sort:");
       int[] selection = array.clone();
       selectionSort(selection);
       printArray(selection);
 
       System.out.println("Insertion Sort:");
       int[] insertion = array.clone();
       insertionSort(insertion);
       printArray(insertion);
   }
}
 
 
✅ 2. Divide and Conquer (Merge Sort & Quick Sort)
// DivideAndConquer.java
public class DivideAndConquer {
 
   public static void mergeSort(int[] arr, int left, int right) {
       if (left < right) {
           int mid = (left + right) / 2;
           mergeSort(arr, left, mid);
           mergeSort(arr, mid + 1, right);
           merge(arr, left, mid, right);
       }
   }
 
   public static void merge(int[] arr, int l, int m, int r) {
       int n1 = m - l + 1, n2 = r - m;
       int[] L = new int[n1], R = new int[n2];
       for (int i = 0; i < n1; i++) L[i] = arr[l + i];
       for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];
 
       int i = 0, j = 0, k = l;
       while (i < n1 && j < n2)
           arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
       while (i < n1) arr[k++] = L[i++];
       while (j < n2) arr[k++] = R[j++];
   }
 
   public static int partition(int[] arr, int low, int high) {
       int pivot = arr[high], i = low - 1;
       for (int j = low; j < high; j++)
           if (arr[j] <= pivot) {
               i++;
               int temp = arr[i]; arr[i] = arr[j]; arr[j] = temp;
           }
       int temp = arr[i + 1]; arr[i + 1] = arr[high]; arr[high] = temp;
       return i + 1;
   }
 
   public static void quickSort(int[] arr, int low, int high) {
       if (low < high) {
           int pi = partition(arr, low, high);
           quickSort(arr, low, pi - 1); quickSort(arr, pi + 1, high);
       }
   }
 
   public static void print(int[] arr) {
       for (int i : arr) System.out.print(i + " ");
       System.out.println();
   }
 
   public static void main(String[] args) {
       int[] arr = {10, 7, 8, 9, 1, 5};
       mergeSort(arr.clone(), 0, arr.length - 1);
       quickSort(arr.clone(), 0, arr.length - 1);
   }
}
 
✅ 3. Fractional Knapsack & Job Sequencing
// GreedyProblems.java
import java.util.*;
 
class Item {
   int value, weight;
   Item(int v, int w) { value = v; weight = w; }
}
 
public class GreedyProblems {
 
   public static double fractionalKnapsack(Item[] items, int W) {
       Arrays.sort(items, (a, b) -> Double.compare((double)b.value / b.weight, (double)a.value / a.weight));
       double totalValue = 0.0;
       for (Item item : items) {
           if (W >= item.weight) {
               totalValue += item.value; W -= item.weight;
           } else {
               totalValue += ((double)item.value / item.weight) * W; break;
           }
       }
       return totalValue;
   }
 
   static class Job {
       char id; int deadline, profit;
       Job(char id, int deadline, int profit) { this.id = id; this.deadline = deadline; this.profit = profit; }
   }
 
   public static void jobSequencing(Job[] jobs) {
       Arrays.sort(jobs, (a, b) -> b.profit - a.profit);
       boolean[] slot = new boolean[jobs.length];
       char[] result = new char[jobs.length];
       for (Job job : jobs) {
           for (int j = Math.min(jobs.length - 1, job.deadline - 1); j >= 0; j--) {
               if (!slot[j]) {
                   slot[j] = true; result[j] = job.id; break;
               }
           }
       }
       for (char c : result) if (c != '\0') System.out.print(c + " ");
   }
 
   public static void main(String[] args) {
       Item[] items = { new Item(60, 10), new Item(100, 20), new Item(120, 30) };
       System.out.println(fractionalKnapsack(items, 50));
 
       Job[] jobs = { new Job('a', 2, 100), new Job('b', 1, 19), new Job('c', 2, 27) };
       jobSequencing(jobs);
   }
}
 
 
✅ 4. Dijkstra Algorithm using Greedy
// Dijkstra.java
import java.util.*;
 
public class Dijkstra {
   static int V = 5;
 
   static int minDistance(int[] dist, boolean[] sptSet) {
       int min = Integer.MAX_VALUE, min_index = -1;
       for (int v = 0; v < V; v++)
           if (!sptSet[v] && dist[v] <= min) {
               min = dist[v]; min_index = v;
           }
       return min_index;
   }
 
   static void dijkstra(int[][] graph, int src) {
       int[] dist = new int[V]; boolean[] sptSet = new boolean[V];
       Arrays.fill(dist, Integer.MAX_VALUE); dist[src] = 0;
 
       for (int count = 0; count < V - 1; count++) {
           int u = minDistance(dist, sptSet); sptSet[u] = true;
           for (int v = 0; v < V; v++)
               if (!sptSet[v] && graph[u][v] != 0 && dist[u] != Integer.MAX_VALUE
                       && dist[u] + graph[u][v] < dist[v])
                   dist[v] = dist[u] + graph[u][v];
       }
 
       for (int i = 0; i < V; i++) System.out.println("Distance from " + src + " to " + i + ": " + dist[i]);
   }
 
   public static void main(String[] args) {
       int[][] graph = {
           {0, 10, 0, 30, 100},
           {10, 0, 50, 0, 0},
           {0, 50, 0, 20, 10},
           {30, 0, 20, 0, 60},
           {100, 0, 10, 60, 0}
       };
       dijkstra(graph, 0);
   }
}
 
✅ 5. Minimum Cost Spanning Tree (Prim's & Kruskal's)
// PrimsAndKruskals.java
import java.util.*;
 
public class PrimsAndKruskals {
 
   static int V = 5;
 
   public static void prims(int[][] graph) {
       int[] key = new int[V]; boolean[] mstSet = new boolean[V]; int[] parent = new int[V];
       Arrays.fill(key, Integer.MAX_VALUE); key[0] = 0; parent[0] = -1;
 
       for (int count = 0; count < V - 1; count++) {
           int u = -1, min = Integer.MAX_VALUE;
           for (int v = 0; v < V; v++) if (!mstSet[v] && key[v] < min) { min = key[v]; u = v; }
           mstSet[u] = true;
           for (int v = 0; v < V; v++)
               if (graph[u][v] != 0 && !mstSet[v] && graph[u][v] < key[v]) {
                   parent[v] = u; key[v] = graph[u][v];
               }
       }
 
       for (int i = 1; i < V; i++) System.out.println(parent[i] + " - " + i + ": " + graph[i][parent[i]]);
   }
 
   static class Edge implements Comparable<Edge> {
       int src, dest, weight;
       public Edge(int s, int d, int w) { src = s; dest = d; weight = w; }
       public int compareTo(Edge compareEdge) { return this.weight - compareEdge.weight; }
   }
 
   static class Subset {
       int parent, rank;
   }
 
   static int find(Subset[] subsets, int i) {
       if (subsets[i].parent != i) subsets[i].parent = find(subsets, subsets[i].parent);
       return subsets[i].parent;
   }
 
   static void union(Subset[] subsets, int x, int y) {
       int xroot = find(subsets, x), yroot = find(subsets, y);
       if (subsets[xroot].rank < subsets[yroot].rank)
           subsets[xroot].parent = yroot;
       else if (subsets[xroot].rank > subsets[yroot].rank)
           subsets[yroot].parent = xroot;
       else {
           subsets[yroot].parent = xroot; subsets[xroot].rank++;
       }
   }
 
   public static void kruskal() {
       int V = 4;
       Edge[] edges = {
           new Edge(0, 1, 10), new Edge(0, 2, 6), new Edge(0, 3, 5), new Edge(1, 3, 15), new Edge(2, 3, 4)
       };
 
       Arrays.sort(edges);
       Subset[] subsets = new Subset[V];
       for (int i = 0; i < V; i++) { subsets[i] = new Subset(); subsets[i].parent = i; subsets[i].rank = 0; }
 
       for (Edge e : edges) {
           int x = find(subsets, e.src), y = find(subsets, e.dest);
           if (x != y) {
               System.out.println(e.src + " - " + e.dest + ": " + e.weight);
               union(subsets, x, y);
           }
       }
   }
 
   public static void main(String[] args) {
       int[][] graph = {
           {0, 2, 0, 6, 0},
           {2, 0, 3, 8, 5},
           {0, 3, 0, 0, 7},
           {6, 8, 0, 0, 9},
           {0, 5, 7, 9, 0}
       };
       prims(graph);
       kruskal();
   }
}
 
✅ 6. Bellman-Ford and Floyd-Warshall
// BellmanFordAndFloyd.java
import java.util.*;
 
public class BellmanFordAndFloyd {
 
   public static void bellmanFord(int[][] edges, int V, int src) {
       int[] dist = new int[V]; Arrays.fill(dist, Integer.MAX_VALUE); dist[src] = 0;
       for (int i = 1; i < V; i++)
           for (int[] edge : edges)
               if (dist[edge[0]] != Integer.MAX_VALUE && dist[edge[0]] + edge[2] < dist[edge[1]])
                   dist[edge[1]] = dist[edge[0]] + edge[2];
 
       System.out.println("Bellman-Ford:");
       for (int i = 0; i < V; i++) System.out.println("To " + i + ": " + dist[i]);
   }
 
   public static void floydWarshall(int[][] graph) {
       int V = graph.length;
       int[][] dist = new int[V][V];
       for (int i = 0; i < V; i++)
           for (int j = 0; j < V; j++)
               dist[i][j] = graph[i][j];
 
       for (int k = 0; k < V; k++)
           for (int i = 0; i < V; i++)
               for (int j = 0; j < V; j++)
                   if (dist[i][k] + dist[k][j] < dist[i][j])
                       dist[i][j] = dist[i][k] + dist[k][j];
 
       System.out.println("Floyd-Warshall:");
       for (int[] row : dist) System.out.println(Arrays.toString(row));
   }
 
   public static void main(String[] args) {
       int[][] edges = {{0, 1, -1}, {0, 2, 4}, {1, 2, 3}, {1, 3, 2}, {1, 4, 2}, {3, 2, 5}, {3, 1, 1}, {4, 3, -3}};
       bellmanFord(edges, 5, 0);
 
       int[][] fwGraph = {
           {0, 3, Integer.MAX_VALUE, 5},
           {2, 0, Integer.MAX_VALUE, 4},
           {Integer.MAX_VALUE, 1, 0, Integer.MAX_VALUE},
           {Integer.MAX_VALUE, Integer.MAX_VALUE, 2, 0}
       };
       floydWarshall(fwGraph);
   }
}
 
 
✅ 7. Longest Common Subsequence using Dynamic Programming
// LCS.java
public class LCS {
 
   static int lcs(String X, String Y) {
       int m = X.length(), n = Y.length();
       int[][] dp = new int[m + 1][n + 1];
 
       // Build the LCS table
       for (int i = 0; i <= m; i++) {
           for (int j = 0; j <= n; j++) {
               if (i == 0 || j == 0)
                   dp[i][j] = 0;
               else if (X.charAt(i - 1) == Y.charAt(j - 1))
                   dp[i][j] = dp[i - 1][j - 1] + 1;
               else
                   dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
           }
       }
 
       return dp[m][n];
   }
 
   public static void main(String[] args) {
       String s1 = "AGGTAB";
       String s2 = "GXTXAYB";
       System.out.println("Length of LCS: " + lcs(s1, s2));
   }
}
🧠 Explanation:
• Builds a table comparing substrings of both strings.
• If characters match, take diagonal + 1.
• Else, take max of left or top cell.
• Bottom-right cell gives final LCS length.
 
✅ 8. Sum of Subset using Backtracking
// SubsetSumBacktrack.java
public class SubsetSumBacktrack {
 
   static void subsetSum(int[] arr, int index, int target, String current) {
       if (target == 0) {
           System.out.println("Subset: " + current);
           return;
       }
 
       if (index == arr.length || target < 0)
           return;
 
       // Include current element
       subsetSum(arr, index + 1, target - arr[index], current + arr[index] + " ");
 
       // Exclude current element
       subsetSum(arr, index + 1, target, current);
   }
 
   public static void main(String[] args) {
       int[] arr = {3, 1, 2, 5};
       int target = 6;
       subsetSum(arr, 0, target, "");
   }
}
🧠 Explanation:
• Recursively includes/excludes elements to find subsets adding up to target.
 
✅ Graph Coloring using Backtracking
// GraphColoring.java
public class GraphColoring {
 
   static boolean isSafe(int v, int[][] graph, int[] color, int c) {
       for (int i = 0; i < graph.length; i++)
           if (graph[v][i] == 1 && color[i] == c)
               return false;
       return true;
   }
 
   static boolean solve(int[][] graph, int m, int[] color, int v) {
       if (v == graph.length)
           return true;
 
       for (int c = 1; c <= m; c++) {
           if (isSafe(v, graph, color, c)) {
               color[v] = c;
               if (solve(graph, m, color, v + 1))
                   return true;
               color[v] = 0;
           }
       }
 
       return false;
   }
 
   public static void main(String[] args) {
       int[][] graph = {
           {0, 1, 1, 1},
           {1, 0, 1, 0},
           {1, 1, 0, 1},
           {1, 0, 1, 0}
       };
       int m = 3; // max 3 colors
       int[] color = new int[graph.length];
 
       if (solve(graph, m, color, 0)) {
           for (int c : color)
               System.out.print(c + " ");
       } else {
           System.out.println("No solution");
       }
   }
}
🧠 Explanation:
• Tries coloring each vertex.
• Backtracks if two connected vertices get same color.
 
✅ 9. Euclidean & Jaccard Distance
// Distance.java
import java.util.*;
 
public class Distance {
 
   public static double euclidean(double[] a, double[] b) {
       double sum = 0;
       for (int i = 0; i < a.length; i++)
           sum += Math.pow(a[i] - b[i], 2);
       return Math.sqrt(sum);
   }
 
   public static double jaccard(Set<String> a, Set<String> b) {
       Set<String> intersection = new HashSet<>(a);
       intersection.retainAll(b);
       Set<String> union = new HashSet<>(a);
       union.addAll(b);
       return (double) intersection.size() / union.size();
   }
 
   public static void main(String[] args) {
       double[] p1 = {1, 2, 3};
       double[] p2 = {4, 5, 6};
       System.out.println("Euclidean Distance: " + euclidean(p1, p2));
 
       Set<String> set1 = new HashSet<>(Arrays.asList("a", "b", "c"));
       Set<String> set2 = new HashSet<>(Arrays.asList("b", "c", "d"));
       System.out.println("Jaccard Similarity: " + jaccard(set1, set2));
   }
}
🧠 Explanation:
• Euclidean: Straight line distance between points.
• Jaccard: Measures overlap in two sets (intersection/union).
 
✅ 10. Bloom Filter in Java
// BloomFilter.java
import java.util.*;
 
public class BloomFilter {
 
   BitSet bitSet;
   int size;
   int[] hashSeeds;
 
   public BloomFilter(int size, int[] seeds) {
       this.size = size;
       this.hashSeeds = seeds;
       this.bitSet = new BitSet(size);
   }
 
   private int getHash(String key, int seed) {
       int hash = 0;
       for (char c : key.toCharArray()) {
           hash = (hash * seed + c) % size;
       }
       return Math.abs(hash);
   }
 
   public void add(String key) {
       for (int seed : hashSeeds)
           bitSet.set(getHash(key, seed));
   }
 
   public boolean mightContain(String key) {
       for (int seed : hashSeeds)
           if (!bitSet.get(getHash(key, seed)))
               return false;
       return true; // might be false positive
   }
 
   public static void main(String[] args) {
       BloomFilter bf = new BloomFilter(100, new int[]{7, 11, 13});
       bf.add("cat");
       bf.add("dog");
 
       System.out.println("cat? " + bf.mightContain("cat"));  // true
       System.out.println("dog? " + bf.mightContain("dog"));  // true
       System.out.println("rat? " + bf.mightContain("rat"));  // maybe false
   }
}
