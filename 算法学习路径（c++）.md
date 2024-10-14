# 算法学习路径（c++）

对于算法学习开始有了一定的了解，初学者应当先具备将自然语言转换成编程语言的能力。在算法学习之初应当慢节奏，前两百题尽量都使用暴力尝试，不要考虑复杂度，再进行技巧尝试进行优化复杂度，这句话很重要。

```c++
ranges::sort(array, graded<>)//降序排列
```

# 哈希表 

## 常见的三种哈希结构

当我们想使用哈希法来解决问题的时候，我们一般会选择如下三种数据结构。

- 数组
- set （集合）
- map(映射)

在C++中，set 和 map 分别提供以下三种数据结构，其底层实现以及优劣如下表所示：

| 集合          | 底层实现 | 是否有序 | 数值是否可以重复 | 能否更改数值 | 查询效率 | 增删效率 |
| ------------- | -------- | -------- | ---------------- | ------------ | -------- | -------- |
| set           | 红黑树   | 值有序   | 值不可重复       | 值不可修改   | O(log n) | O(log n) |
| multiset      | 红黑树   | 值有序   | 值可重复         | 值不可修改   | O(logn)  | O(logn)  |
| unordered_set | 哈希表   | 值无序   | 值不可重复       | 值不可修改   | O(1)     | O(1)     |

​	unordered_set底层实现为哈希表，set 和multiset 的底层实现是红黑树，红黑树是一种平衡二叉搜索树，所以key值是有序的，但key不可以修改，改动key值会导致整棵树的错乱，所以只能删除和增加。

| 映射          | 底层实现 | 是否有序 | 数值是否可以重复 | 能否更改数值 | 查询效率 | 增删效率 |
| ------------- | -------- | -------- | ---------------- | ------------ | -------- | -------- |
| map           | 红黑树   | key有序  | key不可重复      | key不可修改  | O(logn)  | O(logn)  |
| multimap      | 红黑树   | key有序  | key可重复        | key不可修改  | O(log n) | O(log n) |
| unordered_map | 哈希表   | key无序  | key不可重复      | key不可修改  | O(1)     | O(1)     |

## [706. 设计哈希映射](https://leetcode.cn/problems/design-hashmap/)（哈希构建）

设计哈希映射有两种方法：1、链地址法 2、构建大容量数组存储

```c++
//构建数组法
class MyHashMap {
public:
    vector<int>map;
    MyHashMap() {
        const int N = 1000001;
        map = vector<int>(N, -1);
    }
    
    void put(int key, int value) {
        map[key] = value;
    }
    
    int get(int key) {
        return map[key];
    }
    
    void remove(int key) {
        map[key] = -1;
    }
};

//2、链式地址法
class MyHashMap {
private:
    vector<list<pair<int, int>>> data;
    static const int base = 769;
    static int hash(int key) {
        return key % base;
    }
public:
    MyHashMap(): data(base) {}
    
    void put(int key, int value) {
        int h = hash(key);
        for (auto it = data[h].begin(); it != data[h].end(); it++) {
            if ((*it).first == key) {
                (*it).second = value;
                return;
            }
        }
        data[h].push_back(make_pair(key, value));
    }
    
    int get(int key) {
        int h = hash(key);
        for (auto it = data[h].begin(); it != data[h].end(); it++) {
            if ((*it).first == key) {
                return (*it).second;
            }
        }
        return -1;
    }
    
    void remove(int key) {
        int h = hash(key);
        for (auto it = data[h].begin(); it != data[h].end(); it++) {
            if ((*it).first == key) {
                data[h].erase(it);
                return;
            }
        }
    }
};
```

## [3121. 统计特殊字母的数量 II](https://leetcode.cn/problems/count-the-number-of-special-characters-ii/)

```c++
class Solution {
public:
    int numberOfSpecialChars(string word) {
        int n = word.size(), ans = 0;
        unordered_map<char, int>low, upp;//low小写，uop大写
        for(int i = 0; i < n; i++){
            if(word[i] <= 'Z' && upp.count(word[i]) == 0){//寻找大写字母第一次出现的位置
                upp[word[i]] = i;
            }else low[word[i]] = i;
        }
        //low.count(x.first + 32) == 1说明大小写都已经出现了，x.first + 32将大写转换为小写
        //low[x.first + 32] < x.second比较出现位置
        for(auto x : upp){
            if(low.count(x.first + 32) == 1 && low[x.first + 32] < x.second) ans++;
        }
        return ans;
    }
};
```

# 分组循环

## 思路

按照题目要求，数组会被分割成若干组，且每一组的判断/处理逻辑是一样的。

## 核心思想

外层循环负责遍历组之前的准备工作（记录开始位置），和遍历组之后的统计工作（更新答案最大值）。
内层循环负责遍历组，找出这一组最远在哪结束。

## [2760. 最长奇偶子数组](https://leetcode.cn/problems/longest-even-odd-subarray-with-threshold/)

```c++
class Solution {
public:
    int longestAlternatingSubarray(vector<int>& nums, int threshold) {
        int i = 0, ans = 0;
        while(i < nums.size()){
            if(nums[i] % 2 != 0 || nums[i] > threshold){//寻找符合起始位置要求的下标
                i++;
                continue;
            }
            int start = i;//找到了符合要求的起始位置
            i++;//将下标移动到后一位方便后续进行判断
            while(i < nums.size() && nums[i] % 2 != nums[i - 1] % 2 && nums[i] <= threshold){//判断符合要求的连续子数组
                i++;
            }
            ans = max(ans, i - start);//更新最长数组
        }
        return ans;
    }
};
```

## [2765. 最长交替子数组](https://leetcode.cn/problems/longest-alternating-subarray/)

```c++
class Solution {
public:
    int alternatingSubarray(vector<int>& nums) {
        int res = -1, n = nums.size(), i = 0;
        while(i < n - 1){//外层循环遍历数组，记录位置然后更新答案
            if(nums[i + 1] - nums[i] != 1){//不符合s1 = s0 + 1 的要求
                i++;
                continue;//跳过执行下次操作
            }
            int start = i;
            i += 2;//向右偏移两位
            while(i < n && nums[i] == nums[i - 2]){//内层循环遍历区间内的元素
                i++;
            }
            res = max(res, i - start);
            i--;
        }
        return res;
    }
};
```

## [1957. 删除字符使字符串变好](https://leetcode.cn/problems/delete-characters-to-make-fancy-string/)

```c++
class Solution {  
public:  
    string makeFancyString(string s) {  
        string ans;  
        int n = s.size();  
        for (int i = 0; i < n;) {  
            int start = i;  
            while (i < n && s[i] == s[start]) {  
                i++;  
            }  
            int cnt = min(i - start, 2);  
            ans += s.substr(start, cnt);  
        }  
        return ans;  
    }  
};
```

## [2038. 如果相邻两个颜色均相同则删除当前颜色](https://leetcode.cn/problems/remove-colored-pieces-if-both-neighbors-are-the-same-color/)

```c++
class Solution {
public:
    bool winnerOfGame(string colors) {
        int n = colors.size(), i = 0, alice = 0, bob = 0;
        while(i < n){
            int start = i++;
            while(i < n && colors[i] == colors[start]){
                i++;
            }
            int cnt = max(0, i - start - 2);//错误点1：cnt = max(cnt, i - start - 2)
            if(colors[start] == 'A') alice += cnt;
            else bob += cnt;
        }
        return alice > bob;
    }
};
//错误点1：变成了和之前的数值对比并更新，而题意应当是每次都重新从0开始比较并更新需要删除的数量
```

## [228. 汇总区间](https://leetcode.cn/problems/summary-ranges/)

```c++
class Solution {
public:
    vector<string> summaryRanges(vector<int>& nums) {
        int n = nums.size(), i = 0;
        vector<string>ans;
        while(i < n){
            //为何不能使用i++，因为改变了首个位置无法正确与后面元素比较，如：0，2，3那么0->3是不正确的
            int start = i;
            while(i < n - 1 && nums[i] + 1 == nums[i + 1]){
                i++;
            }
            if(start == i) ans.push_back(to_string(nums[start]));//如果区间只包含一个元素
            else ans.push_back(to_string(nums[start]) + "->" + to_string(nums[i]));
            i++;
        }
        return ans;
    }   
};
```

# 前缀和

为何要构造一个前置0，是因为可以避免边界判断，随拿随取。

## [2559. 统计范围内的元音字符串数](https://leetcode.cn/problems/count-vowel-strings-in-ranges/) 1435

```c++
class Solution {
public:
    vector<int> vowelStrings(vector<string>& words, vector<vector<int>>& queries) {
        unordered_set<char>vis = {'a', 'e', 'i', 'o', 'u'};
        int n = words.size();
        int s[n + 1];
        s[0] = 0;//第一个元素为0，方便后续统计前缀和
        for(int i = 0; i < n; i++){
            char a = words[i][0], b = words[i].back();//字符串的首个字符，最后一个字符
            s[i + 1] = s[i] + (vis.count(a) && vis.count(b));//首尾字符都是元音字母，为真则加1
        }
        vector<int>ans;//结果集
        for(auto &q : queries){
            int l = q[0], r = q[1];//划分查询范围
            ans.push_back(s[r + 1] - s[l]);//统计前缀和
        } 
        return ans;
    }
};
```

## [560. 和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/)

```c++
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int sum = 0, ans = 0;
        unordered_map<int,int>map;
        map[0] = 1;
        for(auto x : nums){
            sum += x;
            if(map.find(sum - k) != map.end()) ans += map[sum - k];
            map[sum]++;
        }
        return ans;
    }
};
```

## [1124. 表现良好的最长时间段](https://leetcode.cn/problems/longest-well-performing-interval/) 1908

```c++
class Solution {
public:
    int longestWPI(vector<int>& hours) {
        unordered_map<int,int>map;
        int n = hours.size(), sum = 0, ans = 0;
        for(int i = 0; i < n; i++){
            sum += hours[i] > 8 ? 1 : -1;
            if(sum > 0){
                ans = i + 1;
            }else if(map.count(sum - 1)){
                ans = max(ans, i - map[sum - 1]);
            }
            if(!map.count(sum)){
                map[sum] = i;
            }
        }
        return ans;
    }
};
```

## [2438. 二的幂数组中查询范围内的乘积](https://leetcode.cn/problems/range-product-queries-of-powers/) 1610

```c++
const int MOD = 1000000007;   
class Solution {  
public:  
    vector<int> productQueries(int n, vector<vector<int>>& queries) {  
        vector<int> a;  
        while (n > 0) {  
            int lb = n & -n; // 找到最低位的1  
            a.push_back(lb);  
            n ^= lb; // 清除最低位的1  
        }  
          
        vector<int>ans;  
        for (const auto& query : queries) {  
            int l = query[0];  
            int r = query[1];  
            int product = accumulate(a.begin() + l, a.begin() + r + 1, 1,   
                [](int x, int y) { return (long long)x * y % MOD; } // 使用long long防止溢出  
            );  
            ans.push_back(product);  
        }  
        return ans;  
    }  
};
```

## [1685. 有序数组中差绝对值之和](https://leetcode.cn/problems/sum-of-absolute-differences-in-a-sorted-array/) 1496

```c++
对于i的左边：
nums[i]必然大于左边的所有元素，所以可以数出左边的元素个数，即i个
result[i] = (nums[i] - nums[0]) + (nums[i] - nums[1]) + ......
                                        +(nums[i] - nums[i - 1])
合并后就为:
result[i] = i * nums[i] - (nums[0] + nums[1] + ... + nums[i - 1])
          = i * nums[i] - leftsum
    
    
class Solution {
public:
    vector<int> getSumAbsoluteDifferences(vector<int>& nums) {
        int n = nums.size(),leftsum = 0,rightsum = accumulate(nums.begin(),nums.end(),0);
        vector<int> result(n);
        for(int i = 0; i < n ; i++){
            rightsum -= nums[i];
            result[i] = i * nums[i] - leftsum + rightsum - (n - i - 1) * nums[i];
            leftsum += nums[i];
        }
        return result;
    }
};
```



# 滑动窗口

## 思路：

如何维护窗口和处理边界情况很重要！！！

## 定长滑动窗口

定长滑动窗口题型一般都会给出切割子数组的长度，根据长度进行窗口的滑动，一般来说都是加入一个，减去一个，再更新数值即可，**定长滑动窗口的i一般都位于第二个窗口的左边界**，如：

```c++
for(int i = k; i < n; i++){//k为窗口长度
	if(i >= k){//开始滑动窗口
	......
	}
}
```

### [1456. 定长子串中元音的最大数目](https://leetcode.cn/problems/maximum-number-of-vowels-in-a-substring-of-given-length/) 1263

```c++
class Solution {
public:
    int yuanYing(char c){
        if(c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u'){
            return 1;
        }else return 0;
    }

    int maxVowels(string s, int k) {  
        int n = s.size(), ans = 0, cnt = 0;
        for(int i =0; i < n; i++){
            cnt += yuanYing(s[i]);
            if(i >= k){
                cnt -= yuanYing(s[i - k]);
            }
            ans = max(ans, cnt);
        }
        return ans;
    }
};
```



### [1984. 学生分数的最小差值](https://leetcode.cn/problems/minimum-difference-between-highest-and-lowest-of-k-scores/) 1306

```c++
class Solution {
public:
//思路先排序分出窗口内的边界最大最小值，然后滑动窗口更新最小值
//问题是如何规定这个窗口对我来说是个难题，我不想用同向双指针感觉效率太低
//
    int minimumDifference(vector<int>& nums, int k){
        ranges::sort(nums);
        int ans = nums[k - 1] - nums[0];
        if(nums.size() < k) return 0;
        for(int i = k; i < nums.size(); i++){
            ans = min(ans, nums[i] - nums[i - k + 1]);
        }
        return ans;
    }
};
```

### [643. 子数组最大平均数 I](https://leetcode.cn/problems/maximum-average-subarray-i/)

```c++
class Solution {
public:
    double findMaxAverage(vector<int>& nums, int k) {
        int sum = accumulate(nums.begin(), nums.begin() + k, 0);
        int ans = sum;
        int n = nums.size();
        for(int i = 1; i < n - k + 1; i++){
            sum += nums[i + k - 1] - nums[i - 1];
            ans = max(ans, sum);
        }
        return (double)ans / k;
    }
};
//利用前边第一个子数组的值进行后续的窗口的滑动
```

### [1343. 大小为 K 且平均值大于等于阈值的子数组数目](https://leetcode.cn/problems/number-of-sub-arrays-of-size-k-and-average-greater-than-or-equal-to-threshold/) 1317

```c++
class Solution {
public:
    int numOfSubarrays(vector<int>& arr, int k, int threshold) {
        int n = arr.size(), cnt = 0;
        int sum = accumulate(arr.begin(), arr.begin() + k, 0);
        if(sum / k >= threshold){
            cnt = 1;
        }else cnt = 0;
        for(int i = 1; i < n - k + 1; i++){
            sum += arr[i + k - 1] - arr[i - 1];
            if(sum / k >= threshold) cnt++;
        }
        return cnt;
    }
};
```

### [2090. 半径为 k 的子数组平均值](https://leetcode.cn/problems/k-radius-subarray-averages/) 1358

```c++
class Solution {
public:
    vector<int> getAverages(vector<int>& nums, int k) {
        int n = nums.size();
        int cnt = 2 * k + 1;
        vector<int>avgs(n, -1);  
        if(n < cnt) return avgs;
        long long sum = accumulate(nums.begin(), nums.begin() + cnt, 0L); 
        avgs[k++] = sum / cnt;
        for(int i = cnt; i < n; i++){
            sum += nums[i] - nums[i - cnt];
            avgs[k++] = sum / cnt;
        }
        return avgs;
    }
};
```

### [1052. 爱生气的书店老板](https://leetcode.cn/problems/grumpy-bookstore-owner/) 1418

```c++
class Solution {
public:
    int maxSatisfied(vector<int>& customers, vector<int>& grumpy, int minutes) {
        int n = customers.size(), ans = 0;
        for(int i = 0; i < n; i++){//先累加顾客满意的，并将顾客满意的值置为0
            if(grumpy[i] == 0){
                ans += customers[i];
                customers[i] = 0;
            }
        }
        int cnt = 0, sum = 0;
        for(int i = 0; i < n; i++){//开始滑动窗口剩下不满意的，以不生气技巧为窗口长度开始滑动
            sum += customers[i];
            if(i >= minutes) sum -= customers[i - minutes];
            cnt = max(cnt, sum);
        }
        return ans + cnt;
    }
};
```

### [2379. 得到 K 个黑块的最少涂色次数](https://leetcode.cn/problems/minimum-recolors-to-get-k-consecutive-black-blocks/) 1360

```c++
class Solution {
public:
    int minimumRecolors(string blocks, int k) {
        int n = blocks.size();
        int sum = count(blocks.begin(), blocks.begin() + k, 'W');
        int cnt = sum;
        for(int i = k; i < n; i++){
            sum += blocks[i] == 'W';
            sum -= blocks[i - k] == 'W';     
            cnt = min(cnt, sum);
        }
        return cnt;
    }
};



//错误代码，不知道为何改变数值，将B改为1，W改为0却不行
class Solution {
public:
    int minimumRecolors(string blocks, int k) {
        int n = blocks.size(), sum = 0, cnt = 0;
        for (int i = 0; i < n; i++) {
            if (blocks[i] == 'W') {
                blocks[i] = 1;
            } else if (blocks[i] == 'B') {
                blocks[i] = 0;
            }
            sum += blocks[i];
            if (i >= k) {
                sum += blocks[i] - blocks[i - k];
            }
            cnt = min(cnt, sum);
        }
        cout << cnt;
        return cnt;
    }
};
```

### [2841. 几乎唯一子数组的最大和](https://leetcode.cn/problems/maximum-sum-of-almost-unique-subarray/) 1546

```c++
class Solution {
public:
    long long maxSum(vector<int>& nums, int m, int k) {
        long long ans = 0, n = nums.size(), sum = 0;
        unordered_map<int,int>map;//记录窗口中不同的元素
        for(int i = 0; i < n; i++){
            sum += nums[i];
            map[nums[i]]++;//保存元素在map中的值
            if(i >= k - 1){//开始滑动窗口     
                if(map.size() >= m) ans = max(ans, sum);//符合条件更新最大值
                sum -= nums[i - k + 1];
                map[nums[i - k + 1]]--;
                if(map[nums[i - k + 1]] == 0) map.erase(nums[i - k + 1]);
            }    
        }
        return ans;
    }
};
//这里我套用了书店老板的边界处理，结果行不通，发现还是要根据题目的要求或者是示例，进行处理边界情况更好一些，现场推出边界的维护
```

### [2461. 长度为 K 子数组中的最大和](https://leetcode.cn/problems/maximum-sum-of-distinct-subarrays-with-length-k/) 1553

```c++
class Solution {
public:
    long long maximumSubarraySum(vector<int>& nums, int k) {
        long long n = nums.size(), sum = 0, ans = 0;
        unordered_map<int,int>map;
        if(n < k) return 0;
        for(int i = 0; i < n; i++){
            sum += nums[i];
            map[nums[i]]++;
            if(i >= k - 1){
                if(map.size() == k){
                    ans = max(ans, sum);
                    sum -= nums[i - k + 1];
                }else if(map.size() < k){
                    sum -= nums[i - k + 1];
                } 
                if(--map[nums[i - k + 1]] == 0) map.erase(nums[i - k + 1]);
            }
            
        }
        return ans;
    }
};
```

### [1423. 可获得的最大点数](https://leetcode.cn/problems/maximum-points-you-can-obtain-from-cards/) 1574

```c++
class Solution {
public:
    int maxScore(vector<int> &cardPoints, int k) {
        int n = cardPoints.size();
        int m = n - k;
        int s = accumulate(cardPoints.begin(), cardPoints.begin() + m, 0);
        int min_s = s;
        for (int i = m; i < n; i++) {
            s += cardPoints[i] - cardPoints[i - m];
            min_s = min(min_s, s);
        }
        return accumulate(cardPoints.begin(), cardPoints.end(), 0) - min_s;
    }
};
//正难则反，去掉k张牌就变成了寻找一个滑窗的最小值之和，用总和减去就可得到k张牌的最大值
```

## 不定长滑窗

**思路：**不定长滑窗几乎都是在循环里面加入**while循环**进行特判并压缩窗口。

如果遇到计算子数组个数，并且**子数组元素可重复**的，就可以以它为中心，统计右边的数量，再压缩窗口。

### [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s){
        int n = s.length(), ans = 0, left = 0;
        unordered_set<char> window;//用unordered_set是因为查删时间复杂度为O（1）
        for (int right = 0; right < n; right++) {
            char c = s[right];//保存右指针指向的元素
            while (window.count(c)){//查找窗口内是否有重复出现的元素
                window.erase(s[left++]);//将它去除，left右移
            }
            window.insert(c);//没有则插入窗口
            ans = max(ans, right - left + 1);//更新结果
        }
        return ans;
    }
};
//不定长似乎都是用while进行遍历，如果题目要求就可加入特判
```

### [1493. 删掉一个元素以后全为 1 的最长子数组](https://leetcode.cn/problems/longest-subarray-of-1s-after-deleting-one-element/) 1423

```c++
class Solution {
public:
    int longestSubarray(vector<int>& nums) {
        int n = nums.size(), l = 0, r = 0, ans = 0, cout = 0;
        while(r < n){
            cout += nums[r] == 0;
            while(cout > 1) cout -= nums[l++] == 0;
            ans = max(ans, r - l + 1);
            r++;
        }
        return ans - 1;
    }
};
```



### [2730. 找到最长的半重复子字符串](https://leetcode.cn/problems/find-the-longest-semi-repetitive-substring/) 1502

```c++
class Solution {
public:
    int longestSemiRepetitiveSubstring(string s) {
        int n = s.size(), l = 0, cout = 0, ans = 1;
        for(int r = 1; r < n; r++){
            cout += s[r] == s[r - 1];
            while(cout > 1){
                if(s[l] == s[l + 1]){
                    cout--;
                }
                l++;
            }
            ans = max(ans, r - l + 1);
        }
        return ans;
    }
};
//由此可得出不一定要while循环，但是都是用while循环判断窗口元素再进行压缩窗口
```

### [904. 水果成篮](https://leetcode.cn/problems/fruit-into-baskets/) 1516

```c++
class Solution {
public:
    int totalFruit(vector<int>& fruits) {
        int n = fruits.size(), l = 0, r = 0, ans = 0;
        unordered_map<int,int>map;
        for(; r < n; r++){
            map[fruits[r]]++;//统计水果种类的数量
            while(map.size() > 2){//超过两种种类，压缩窗口
                if(--map[fruits[l]] == 0) map.erase(fruits[l]);
                l++;
            }
            ans = max(ans, r - l + 1);
        }
        return ans;
    }
};
```

### [1695. 删除子数组的最大得分](https://leetcode.cn/problems/maximum-erasure-value/) 1529

```c++
class Solution {
public:
    int maximumUniqueSubarray(vector<int>& nums) {
        int n = nums.size(), ans = 0, r = 0, l = 0;
        unordered_map<int,int>map;
        int sum = 0;
        for(int i =0; i < n; i++){
            map[nums[i]]++;
            sum += nums[r];
            while(map.size() != r - l + 1){
                if(--map[nums[l]] == 0) map.erase(nums[l]);
                sum -= nums[l++];
            }
            r++;
            ans = max(ans, sum);
        }
        return ans;
    }
};
```

### [2958. 最多 K 个重复元素的最长子数组](https://leetcode.cn/problems/length-of-longest-subarray-with-at-most-k-frequency/) 1535

```c++
class Solution {
public:
    int maxSubarrayLength(vector<int>& nums, int k) {
        int n = nums.size(), ans = 0, r = 0, l = 0;
        unordered_map<int,int>map;
        for(int r = 0; r < n; r++){
            map[nums[r]]++;
            while(map[nums[r]] > k){
                if(--map[nums[l]] == 0) map.erase(nums[l]);
                l++;
            }
            ans = max(ans, r - l + 1);
        }
        return ans;
    }
};
```

### [2024. 考试的最大困扰度](https://leetcode.cn/problems/maximize-the-confusion-of-an-exam/) 1643

```c++
class Solution {
public:
    int maxConsecutiveAnswers(string answerKey, int k) {
        int l = 0, r = 0, cntF = 0, cntT = 0;

        while(r < answerKey.size()){
            answerKey[r] == 'T' ? cntT++ : cntF++;
            if(k - ((r - l + 1) - max(cntF, cntT)) < 0){
                answerKey[l] == 'T' ? cntT-- : cntF--;
                l++;
            }          
            r++;  
        }
        return r - l;
    }
};
//只需要维护一个字符，保证T/F小于K即可
```

### [1004. 最大连续1的个数 III](https://leetcode.cn/problems/max-consecutive-ones-iii/) 1656

```c++
class Solution {
public:
    int longestOnes(vector<int>& nums, int k) {
        int n = nums.size(), l = 0, r = 0, cnt = 0, ans = 0;
        while(r < n){
            if(nums[r] == 0) cnt++;//统计窗口内0的个数
            while(k < cnt){//如果操作次数不够，压缩窗口
                if(nums[l] == 0){
                    cnt--;  
                }
                l++;   
            }
            ans = max(ans, r- l + 1);
            r++; 
        }
        return ans;
    }
};
//很简单的一题，不像是1650分的题目
```

### [1438. 绝对差不超过限制的最长连续子数组](https://leetcode.cn/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/) 1672

```c++
class Solution {
public:
    int longestSubarray(vector<int>& nums, int limit) {
        int n = nums.size(), l = 0, r = 0, res = 0;
        multiset<int>sv;
        for(r; r < n; r++){
            sv.insert(nums[r]);
            while(*sv.rbegin() - *sv.begin() > limit){//*是解引用，multiset是个迭代器
                sv.erase(sv.find(nums[l]));//
                l++;
            }
            res = max(res, r - l + 1);
        }
        return res;
    }
};
//利用multiset这个结构进行维护区间最大最小值，妙极了！！！
//
```

### [209. 长度最小的子数组](https://leetcode.cn/problems/minimum-size-subarray-sum/)

```c++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int n = nums.size(), sum = 0, l = 0, ans = INT_MAX;
        for(int r = 0; r < n; r++){
            sum += nums[r];
            while(sum >= target){ 
/*这里为何不使用ans = max(ans, r - l + 1);是因为INT_MAX肯定会被更新掉的，无法确认阈值是多少，只能使用三目表达式用来更新窗口大小*/
                ans = ans < r - l + 1 ? ans : r- l + 1;
                sum -= nums[l++];
            }
        }
        return ans == INT_MAX ? 0 : ans;
    }
};
//求最小长度并且要求值大于目标值
```

### [2799. 统计完全子数组的数目](https://leetcode.cn/problems/count-complete-subarrays-in-an-array/) 1398

```c++
class Solution {
public:
    int countCompleteSubarrays(vector<int>& nums) {
        int n = nums.size(), ans = 0;
        int len = unordered_set<int>(nums.begin(), nums.end()).size();
        unordered_map<int,int>map;
        for(int r = 0, l = 0; r < n; r++){
            map[nums[r]]++;
            while(map.size() == len){
                ans += nums.size() - r;//这个思路很重要
                if(--map[nums[l]] == 0) map.erase(nums[l]);
                l++;
            }
        }
        return ans;
    }
};
//输入：nums = [1,3,1,2,2]
//解释：完全子数组有：[1,3,1,2]、[1,3,1,2,2]、[3,1,2] 和 [3,1,2,2]
//先去重求出几个不同元素，方便后续对比，利用map进行记录
//ans += nums.size() - r;遍历到[1,3,1,2]，就已经是一个完全子数组了，以它为中心，枚举还剩下几个完全子数组，用n-r就可得到右边有几个数也能组成完全子数组，再缩减左端窗口即可
```

### [713. 乘积小于 K 的子数组](https://leetcode.cn/problems/subarray-product-less-than-k/)

```c++
class Solution {
public:
    int numSubarrayProductLessThanK(vector<int>& nums, int k) {
        int n = nums.size(), sum = 1, ans = 0;//统计窗口长度即可
        if(k <= 1) return 0;
        for(int r = 0, l = 0; r < n; r++){
            sum *= nums[r]; 
            while(sum >= k){
                sum /= nums[l++];
            }
            ans += r - l + 1;
        }
        return ans;
    }
};
```

### [1358. 包含所有三种字符的子字符串数目](https://leetcode.cn/problems/number-of-substrings-containing-all-three-characters/) 1646

```c++
class Solution {
public:
    int numberOfSubstrings(string s) {
        int n = s.size(), ans = 0;
        unordered_map<char,int>map;
        for(int r = 0, l = 0; r < n; r++){
            map[s[r]]++;
            while(map.size() == 3){
                ans += n - r;
                if(--map[s[l]] == 0) map.erase(s[l]);
                l++;
            }
        }
        return ans;
    }
};
```

### [2962. 统计最大元素出现至少 K 次的子数组](https://leetcode.cn/problems/count-subarrays-where-max-element-appears-at-least-k-times/) 1701

```c++
class Solution {
public:
    long long countSubarrays(vector<int>& nums, int k) {
        int n = nums.size();
        long long ans = 0;
        int x = *max_element(nums.begin(), nums.end());
        unordered_map<int,int>map;
        for(int r = 0, l = 0; r < n; r++){
            map[nums[r]]++;
            while(map[x] >= k){
                ans += n - r;
                if(--map[nums[l]] == 0) map.erase(nums[l]);
                l++;
            }
        }
        return ans;
    }
};
//可以不用map。
class Solution {
public:
    long long countSubarrays(vector<int> &nums, int k) {
        int mx = *max_element(nums.begin(), nums.end());
        long long ans = 0;
        int cnt_mx = 0, left = 0;
        for (int x : nums) {
            cnt_mx += x == mx;
            while (cnt_mx == k) {
                cnt_mx -= nums[left++] == mx;
            }
            ans += left;
        }
        return ans;
    }
};
```

## 多指针滑窗

### [930. 和相同的二元子数组](https://leetcode.cn/problems/binary-subarrays-with-sum/) 1592

```c++
class Solution {
public:
    int numSubarraysWithSum(vector<int>& nums, int goal) {
        int n = nums.size(), sum1 = 0, sum2 = 0, ans = 0;
        for(int r = 0, l1 = 0, l2 = 0; r < n; r++){
            sum1 += nums[r];
            sum2 += nums[r];
            while(sum1 > goal && l1 <= r){
                sum1 -= nums[l1++];
            }
            while(sum2 >= goal && l2 <= r){
                sum2 -= nums[l2++];
            }
            ans += l2 - l1;
        }
        return ans;
    }
};
//l1是固定窗口靠左边的指针，l2是固定窗口靠右边的窗口
//固定窗口右边界寻找左边和相等的两个左边界（一个靠左边，一个靠右边）
```

# 二分查找

二分查找是具有单调性的，但单调性的不一定用二分查找

## 二分搜索

二分查找很多都要动用到数学思维，思维非常跳跃。

```c++
int mid = (r + l) / 2;
int mid = l + (r - l) / 2;//防止溢出，int型为2^32-1
int mid = l + ((r - l)>> 1);//利用了位运算的性质，非负数向右移1位等于除2
```

**思路：**二分查找的前提是有序，寻找**循环不变量**，循环不变量根据查找位置判断，如果你想找到**第一个大于等于**target的元素的位置时`if(nums[mid] < target)`,mid左边区间的元素都是小于target的！！

```
if(nums[mid] < target) left = mid + 1;
else right = mid - 1;
```

查找**第一个大于**target的元素的位置时`if(nums[mid] <= target)`

```
if(nums[mid] <= target) left = mid + 1;
else right = mid - 1;
```

O(logn)时间复杂度

### [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)

```c++
class Solution {
public:
    int getnums(vector<int>& nums, int target){
        int right = (int)nums.size() - 1, left = 0, mid  =0;
        while(left <= right){//闭区间写法
            mid = left + (right - left) / 2;//防止溢出
            if(nums[mid] < target){//如果中间值还是小于目标值，将left左移
                left = mid + 1;
            }else{//反之，right右移
                right = mid - 1;
            }
        }
        return left;//循环最后left在目标值上，right在left左边
    }
    vector<int> searchRange(vector<int>& nums, int target) {
        int start = getnums(nums, target);//
        if(start == nums.size() || nums[start] != target) return {-1, -1};
        int end = getnums(nums, target + 1) - 1;
        return {start, end};
    }
};
```

### [35. 搜索插入位置](https://leetcode.cn/problems/search-insert-position/)

```c++
class Solution {
public:
    int getnums(vector<int>& nums, int target){
        int left = 0, right = nums.size() - 1;
        while(left <= right){
            int mid = left + (right - left) / 2;
            if(nums[mid] < target){
                left = mid + 1;
            }else{
                right = mid - 1;
            }
        }
        return left;
    }
    int searchInsert(vector<int>& nums, int target) {
        int st = getnums(nums, target);
        return st;
    }
};
//自动就会找到插入位置，或目标值索引位置
```

### [704. 二分查找](https://leetcode.cn/problems/binary-search/)

```c++
class Solution {
public:
    int getnums(vector<int>& nums, int target){
        int left = 0, right = nums.size() - 1;
        while(left <= right){
            int mid = left + (right - left) / 2;
            if(nums[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return left;
    }
    int search(vector<int>& nums, int target) {
        int st = getnums(nums, target);
        return st < nums.size() && nums[st] == target ? st : -1;
    }
};
```

### [744. 寻找比目标字母大的最小字母](https://leetcode.cn/problems/find-smallest-letter-greater-than-target/)

```c++
class Solution {
public:
    int getletters(vector<char>& letters, char target){
        int left = 0, right = letters.size() - 1;
        while(left <= right){
            int mid = left + (right - left) / 2;
            if(letters[mid] <= target) left = mid + 1;
            else right = mid - 1;
        }
        return left;
    }
    char nextGreatestLetter(vector<char>& letters, char target) {
        int st = getletters(letters, target);
        return st < letters.size() ? letters[st] : letters[0];
    }
};
```

### [275. H 指数 II](https://leetcode.cn/problems/h-index-ii/)

```c++
class Solution {
public:
    int hIndex(vector<int>& citations) {
        int n = citations.size();
        int l = 0, r = n;
        while (l < r) {
            int mid = l + ((r - l) >> 1);
            if (citations[mid] < n - mid) l = mid + 1;
            else r = mid;
        }
        return n - l;
    }
};
/*这题的左右边界定义又不一样了
r 被初始化为 n 而不是 n-1 的原因是为了确保整个数组都被包含在搜索范围内。然而，这并不意味着 citations[n] 实际上存在，因为在 C++ 中，citations[n] 是不合法的，因为它超出了数组的界限。但是，在二分查找的上下文中，r 仅仅是一个边界标记，而不是一个实际要访问的索引。
```

### [2529. 正整数和负整数的最大计数](https://leetcode.cn/problems/maximum-count-of-positive-integer-and-negative-integer/)

```c++
class Solution {
public:
    //方法1：模拟
    int maximumCount(vector<int>& nums) {
        int pos = 0,neg = 0;
        for(int i = 0; i < nums.size(); i++){
            if(nums[i] > 0)
            {
                pos++;
            }else if(nums[i] < 0)
                neg++;
            }
        return max(pos,neg);
    }
    
    
    //方法2：使用lower_bound && upper_bound语法
    int maximumCount(vector<int>& nums){
        int maximumCount(vector<int> &nums) {
        int neg = ranges::lower_bound(nums, 0) - nums.begin();
        int pos = nums.end() - ranges::upper_bound(nums, 0);
        return max(neg, pos);
    }
    //方法3：使用equal_range语法
    int maximumCount(vector<int> &nums) {
        auto [left, right] = ranges::equal_range(nums, 0);
        int neg = left - nums.begin();
        int pos = nums.end() - right;
        return max(neg, pos);
    }
};
```

### [2300. 咒语和药水的成功对数](https://leetcode.cn/problems/successful-pairs-of-spells-and-potions/) 1477

```c++
class Solution {
public:
    int getnums(vector<int>& potions, double cur){
        int m = potions.size(), l = 0, r = m - 1;
        while(l <= r){
            int mid = l + (r - l) / 2;
            if(potions[mid] < cur) l = mid + 1; 
            else r = mid - 1;
        }
        return m - l;
    }
    vector<int> successfulPairs(vector<int>& spells, vector<int>& potions, long long success) {
        int n = spells.size();
        vector<int>vec(n);
        ranges::sort(potions);
        for(int i = 0; i < n; i++){
            double cur = success * 1.0 / spells[i];
            vec[i] = getnums(potions, cur); 
        }
        return vec;
    }
};
//if(potions[mid] < cur) l = mid + 1;//有时候是<=，精髓还是要找到循环不变量
//else r = mid - 1;
//这题处理数据非常麻烦，对数据处理还是一窍不通
```

### [2389. 和有限的最长子序列](https://leetcode.cn/problems/longest-subsequence-with-limited-sum/)

```c++
class Solution {
public:
    vector<int> answerQueries(vector<int>& nums, vector<int>& queries) {
        int m = queries.size(), n = nums.size();
        ranges::sort(nums);
        for(int i = 1; i < n; i++){
            nums[i] += nums[i - 1];
        }
        for(int &q : queries){
            q = upper_bound(nums.begin(), nums.end(), q) - nums.begin();
        }
        return queries;
    }
};
//为何不能使用lower_bound?
//因为题目中给出了结果数组中的元素都是小于等于的


//此题也可使用贪心
class Solution {
public:
    vector<int> answerQueries(vector<int>& nums, vector<int>& queries) {
        ranges::sort(nums);
        vector<int>vec(queries.size());
        for(int i =0; i < queries.size(); i++){
            int sum = 0;
            for(int j = 0; j < nums.size(); j++){
                sum += nums[j];
                if(sum <= queries[i]){
                    vec[i]++;
                }
            }
        }
        return vec;
    }
};
```

### [1170. 比较字符串最小字母出现频次](https://leetcode.cn/problems/compare-strings-by-frequency-of-the-smallest-character/)

```c++
class Solution {  
public:  
    vector<int> numSmallerByFrequency(vector<string>& queries, vector<string>& words) {  
        // 定义一个lambda函数f，用于计算给定字符串中不同字符的频率  
        auto f = [](string s) {  
            int cnt[26] = {0};  
            for (char c : s) {  
                cnt[c - 'a']++;  
            }  
            for (int x : cnt) {  
                if (x) {  
                    return x;  
                }  
            }  
            return 0;  
        };  
        int n = words.size();  
        // 定义一个整数数组nums，用于存储words中每个字符串的不同字符频率  
        int nums[n];  
        // 遍历words向量  
        for (int i = 0; i < n; i++) {  
            // 调用lambda函数f计算当前字符串的频率，并存储在nums数组中  
            nums[i] = f(words[i]);  
        }   
        sort(nums, nums + n);  
        vector<int> ans;  
        for (auto& q : queries) {  
            // 调用lambda函数f计算当前查询字符串的频率  
            int x = f(q);  
            // 使用upper_bound找到nums数组中第一个大于x的元素的位置  
            // 然后通过位置计算比x大的元素的数量  
            ans.push_back(n - (upper_bound(nums, nums + n, x) - nums));  
        } 
        return ans;  
    }  
};
```

### [2563. 统计公平数对的数目](https://leetcode.cn/problems/count-the-number-of-fair-pairs/) 1721

```c++
class Solution {  
public:  
    long long countFairPairs(vector<int>& nums, int lower, int upper) { 
        int n = nums.size();
        long long ans = 0;  
        ranges::sort(nums);
        for(int j = 0; j < n; j++){  
            // 使用 upper_bound 查找在 nums 的前 j 个元素中，第一个大于 upper - nums[j] 的迭代器位置  
            // 也就是查找所有与 nums[j] 相加后大于 upper 的元素  
            auto r = upper_bound(nums.begin(), nums.begin() + j, upper - nums[j]);  
            
            // 使用 lower_bound 查找在 nums 的前 j 个元素中，第一个不小于 lower - nums[j] 的迭代器位置  
            // 也就是查找所有与 nums[j] 相加后不小于 lower 的元素  
            auto l = lower_bound(nums.begin(), nums.begin() + j, lower - nums[j]);  
            
            // 计算在区间 [l, r) 中元素的数量，即满足 lower <= nums[i] + nums[j] <= upper 的 i 的数量  
            // 这里我们假设 nums[i] 已经在 nums[j] 之前（因为我们在遍历 nums[j]）  
            ans += r - l;  
        }  
        return ans;  
    }  
};
//上面是左闭右开的写法
//下面是左闭右闭的写法

```

### [1146. 快照数组](https://leetcode.cn/problems/snapshot-array/) 1771

```c++
class SnapshotArray{
    int cur_snap_id = 0;
    unordered_map<int, vector<pair<int, int>>> history; // 每个 index 的历史修改记录
public:
    SnapshotArray(int) {}

    void set(int index, int val) {
        history[index].emplace_back(cur_snap_id, val);
    }

    int snap() {
        return cur_snap_id++;
    }

    int get(int index, int snap_id) {
        auto& h = history[index];
        // 找快照编号 <= snap_id 的最后一次修改记录
        // 等价于找快照编号 >= snap_id+1 的第一个修改记录，它的上一个就是答案
        int j = ranges::lower_bound(h, make_pair(snap_id + 1, 0)) - h.begin() - 1;
        return j >= 0 ? h[j].second : 0;
    }
};
//什么时候读得懂题目再来二刷！！！！！
```

## 二分答案求最值

通常要写一个check函数来进行判断！

### [875. 爱吃香蕉的珂珂](https://leetcode.cn/problems/koko-eating-bananas/) 1766

```c++
class Solution {
public:
    long getTime(vector<int>& piles, int k){
        long n = piles.size(), time = 0;
        for(int i = 0; i < n; i++){
            time += (piles[i] + k - 1) / k;//向上取整
        }
        return time;
    }
    int minEatingSpeed(vector<int>& piles, int h) {
        int l = 1, r = *max_element(piles.begin(), piles.end()), k = 0;
        while(l < r){
            k = l + ((r - l) >> 1);
            if(getTime(piles, k) > h) l = k + 1;//向右缩小答案范围
            else r = k;
        }
        return l;
    }
};
```

### [2187. 完成旅途的最少时间](https://leetcode.cn/problems/minimum-time-to-complete-trips/) 1641

```c++
class Solution {
public:
//二分时间
    long long gettime(vector<int>& time, long long mid){
        long long sum = 0;
        for(auto &x : time){
            sum += mid / x;
        }
        return sum;
    }
    long long minimumTime(vector<int>& time, int totalTrips) {
        long long l = 1, max = *max_element(time.begin(), time.end());
        long long r = 1LL * max * totalTrips;
        while(l < r){
            long long mid = l + ((r - l) >> 1);
            if(gettime(time, mid) >= totalTrips) r = mid;//花费时间太多
            else l =  mid + 1;
        }
        return l;
    }
};
//这题难在右边界数据无法确定
```

### [1870. 准时到达的列车最小时速](https://leetcode.cn/problems/minimum-speed-to-arrive-on-time/) 1676

```c++
class Solution {
public:
    double getTime(vector<int>& dist, int mid){
        double sumTime = 0;
        int n = dist.size();
        for(int i = 0; i < n - 1; i++){
            sumTime += (dist[i] + mid - 1) / mid;//需要花费的时间
        }
        sumTime += (double)dist[n - 1] / mid;
        return sumTime;
    }
    int minSpeedOnTime(vector<int>& dist, double hour) {
        int l = 1, r = 1e7 + 1;
        while(l < r){
            int mid = l + ((r - l) >> 1);
            if(getTime(dist, mid) > hour) l = mid + 1;
            else r = mid;
        }
        double sumTime = getTime(dist, l);
        return sumTime > hour ? -1 : l;
    }
};
//二分查找的数据处理太恶心人了
```

### [1011. 在 D 天内送达包裹的能力](https://leetcode.cn/problems/capacity-to-ship-packages-within-d-days/) 1725

```c++
class Solution {
public:
//二分
    int getWeights(vector<int>& weights, int mid){
        int sumTime = 1;
        int cur = 0;
        for(auto &x : weights){
            if(cur + x > mid){
                sumTime++;
                cur = 0;
            }
            cur += x;
        }
        return sumTime;//当前运载量可以完成的天数
    }
    int shipWithinDays(vector<int>& weights, int days) {
        int n = weights.size(), l = *max_element(weights.begin(), weights.end()), r = 1e9 + 1;
        while(l < r){
            int mid = l + ((r - l) >> 1);//二分一天的运载量
            if(getWeights(weights, mid) > days) l = mid + 1; 
            else r = mid;
        }
        return l;
    }
};
/*l的取值是因为包裹无法拆分，最小左边界只能选取数组中最大的包裹重量
```

### [2226. 每个小孩最多能分到多少糖果](https://leetcode.cn/problems/maximum-candies-allocated-to-k-children/) 1646

```c++
class Solution {
public:
    long long getnums(vector<int>& candies, int lim){
        long long sumC = 0;
        for(auto &x : candies){
            sumC += x / lim;
        }
        return sumC;
    }

    int maximumCandies(vector<int>& candies, long long k) {
        long long sum = accumulate(candies.begin(), candies.end(), 0ll);
        if(sum < k) return 0;
        int l = 1, r = 1 + *max_element(candies.begin(), candies.end());//对每个小孩可以得到的最多的糖进行二分
        while(l < r){
            int mid = l + ((r - l) >> 1);
            if(getnums(candies, mid) >= k) l = mid + 1;
            else r = mid;
        }
        return l - 1;
    }
};
/*accumulate，初始为0调用的是int类型，本题初始值要为0LL才行
```

### [100302. 正方形中的最多点数](https://leetcode.cn/problems/maximum-points-inside-the-square/)

```c++
class Solution {
public:
    int getindex(vector<vector<int>>& points, string s, int lim){
        int res = 0;
        bool cnt[26] = {0};
        for(int i = 0; i < s.size(); i++){
            if(abs(points[i][0]) <= lim && abs(points[i][1]) <= lim){
                if(cnt[s[i] - 'a']) return -1;
                res++;
                cnt[s[i] - 'a'] = true;
            }
        }
        return res;
    }
    int maxPointsInsideSquare(vector<vector<int>>& points, string s) {
        int l = 0, r = 1e9 + 1;
        while(l < r){
            int mid = l + ((r - l) >> 1);
            if(getindex(points, s, mid) >= 0) l = mid + 1;
            else r = mid;
        }
        return (l > 0) ? getindex(points, s, l - 1) : 0;
    }
};      
/*二分最大边长，判断正方形内的点是否重复即可
```

 

# 最大公约数、同余原理

**求最大公约数**的递归调用：

gcd(a,b),其中a>b，**时间复杂度**为O((loga)的三次方)

```c++
gcd(a,b)->{return b == 0 ? a : gcd(b, a % b);}
```

**求最小公倍数：**

```c++
return a / gcd(a,b) * b
```



**同余原理：**

加法：

```c++
(a + b) + (c + d) == ((a + b) % MOD + (c + d) % MOD) % MOD
9 mod 3 = 0,4 mod 3 = 1, 5 mod 3 = 2, (1 + 2) mod 3 = 0
```

减法：

```c++
(a - b) % MOD == (a % MOD - b % MOD + MOD) % MOD//处理非负数
```

乘法：需要使用long类型做中间变量，防止整型溢出

```c++
(a * b * c * d) % MOD == ((a * b) % MOD) * ((c * d) % MOD)
```

除法：需要逆元

## [878. 第 N 个神奇数字](https://leetcode.cn/problems/nth-magical-number/)

```c++
class Solution {
    const int MOD = 1e9 + 7;
public:
    int nthMagicalNumber(int n, int a, int b) { 
        long long l = 0, r = (long long)n * min(a,b), ans = 0;
        while(l <= r){
            long long mid = l + ((r - l) >> 1);
            //a / gcd(a,b) * b为最小公倍数
            //mid / a + mid / b - mid / (a / gcd(a,b) * b)求出有几个神奇数字
            if(mid / a + mid / b - mid / (a / gcd(a,b) * b) >= n){
                ans = mid;
                r = mid - 1;
            }else{
                l = mid + 1;
            }
        }
        return ans % MOD;
    }
};
```





# 回溯算法

## 回溯法解决的问题

回溯法，一般可以解决如下几种问题：

- 组合问题：N个数里面按一定规则找出k个数的集合
- 切割问题：一个字符串按一定规则有几种切割方式
- 子集问题：一个N个数的集合里有多少符合条件的子集
- 排列问题：N个数按一定规则全排列，有几种排列方式
- 棋盘问题：N皇后，解数独等等



## 回溯算法模板

```c++
void backtracking(参数) {
    if (终止条件) {
        存放结果;
        return;
    }

    for (选择：本层集合中元素（树中节点孩子的数量就是集合的大小）) {
        处理节点;
        backtracking(路径，选择列表); // 递归
        回溯，撤销处理结果
    }
}
```

## [77. 组合](https://leetcode.cn/problems/combinations/)

```c++
class Solution {  
public:   
    vector<vector<int>> result;    
    vector<int> path;   
    // 定义一个名为backst的递归函数
    void backst(int n, int k, int startindex) {  
        // 如果当前组合的长度等于k，则表示已经构建了一个完整的组合。
        if(path.size() == k) {  
            result.push_back(path);  
            return;  
        }  

        //n-(k - path.size()) + 1进行了剪枝优化
        for(int i = startindex ; i <= n-(k - path.size()) + 1 ; i ++) {  
            path.push_back(i);  
            backst(n, k, i+1); // 递归调用，开始下一个数字的选择  
            path.pop_back(); // 回溯，移除刚刚添加的数字，以便尝试其他选择  
        }  
    }  
    vector<vector<int>> combine(int n, int k) {  
        backst(n , k , 1); // 从索引1开始构建组合，因为索引0不适用（至少需要有一个数字）  
        return result;
    }  
};
```

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        result = []  # 存放结果集
        self.backtracking(n, k, 1, [], result)
        return result
    def backtracking(self, n, k, startIndex, path, result):
        if len(path) == k:
            result.append(path[:])
            return
        for i in range(startIndex, n + 1):
            path.append(i)  # 处理节点
            self.backtracking(n, k, i + 1, path, result)
            path.pop()  # 回溯，撤销处理的节点
```

## [216. 组合总和 III](https://leetcode.cn/problems/combination-sum-iii/)

```c++
class Solution {
public:
    vector<vector<int>>result;
    vector<int>path;
    int sum = 0;
    void backsum(int n, int k, int sum, int startindex){
        if(k == path.size()){
            if(sum == n){
                result.push_back(path);
                return;
            }
        }
        for(int i = startindex; i <= 9 ;i++){
            sum +=i;
            path.push_back(i);
            backsum(n , k , sum , i + 1);
            sum -=i;
            path.pop_back();
        }
    }
    vector<vector<int>> combinationSum3(int k, int n) {
        backsum(n , k , 0 , 1);
        return result;
    }
};
```

> ​		回溯（暴力）其实就是递归，而递归就是循环，在写递归函数中，基本上都是先写代码块，缺啥参数再补上，一般的<u>**剪枝操作**都是在**循环条件**里进行操作。</u>

## [17. 电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)

```c++
class Solution {
public:
    const string lettermap[10] = {
        "",
        "",
        "abc",
        "def",
        "ghi",
        "jkl",
        "mno",
        "pqrs",
        "tuv",
        "wxyz",
    }; 
    vector<string>result;//存放结果集
    string s;//存放叶子节点，暂时满足条件的结果
    //index为lettermap的索引
    void backstr(int index,const string& digits){
        if(index == digits.size()){//满足条件
            result.push_back(s);
            return;
        }
        int digit = digits[index] - '0';//将digits类型转换为int，"23"->23
        string letters = lettermap[digit];
        for(int i = 0; i < letters.size() ; i++){
            s.push_back(letters[i]);
            backstr(index + 1 , digits);
            s.pop_back();
        }
    }
    vector<string> letterCombinations(string digits) {
        s.clear();
        result.clear();
        if(digits.size() == 0) return result;
        backstr(0 , digits);
        return result;
    }
};
```

## [39. 组合总和](https://leetcode.cn/problems/combination-sum/)

```c++
class Solution {
public:
    int sum = 0;
    vector<int>path;
    vector<vector<int>>result;
    void backsum(vector<int>& candidates,int target,int sum,int startindex){
        if(sum > target) return;
        if(sum == target){
            result.push_back(path);
            return;
        }
        for(int i = startindex; i <candidates.size() ; i++){
            sum +=candidates[i];
            path.push_back(candidates[i]);
            backsum(candidates,target, sum , i);
            sum -=candidates[i];
            path.pop_back();
        }
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        result.clear();
        path.clear();
        backsum(candidates,target,0 , 0);
        return result;
    }
};
```

## [40. 组合总和 II](https://leetcode.cn/problems/combination-sum-ii/)

```C++
class Solution {
public:
    vector<int>path;
    vector<vector<int>>result;
    int sum = 0;
    void backsum(vector<int>& candidates, int target,int sum,int startindex,vector<bool>& used){
        if(sum == target){
            result.push_back(path);
            return;
        }
        for(int i = startindex ; i < candidates.size() && sum + candidates[i] <= target; i++){
            if(i > 0 && candidates[i] == candidates[i - 1] && used[i - 1] == false){
                continue;
            }
            sum +=candidates[i];
            path.push_back(candidates[i]);
            used[i] = true;
            backsum(candidates , target , sum , i+1 , used);
            used[i] = false;
            sum -=candidates[i];
            path.pop_back();
        }
    }
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<bool> used(candidates.size(), false);
        result.clear();
        path.clear();
        sort(candidates.begin(), candidates.end());
        backsum(candidates , target , 0 , 0 , used);
        return result;
    }
};
```

## [131. 分割回文串](https://leetcode.cn/problems/palindrome-partitioning/)

```c++
class Solution {
public:
    vector<vector<string>>result;
    vector<string>path;
    void backstr(const string& s,int startindex){
        if(startindex >= s.size()){//切割字符串
            result.push_back(path);
            return;
        }
        for(int i = startindex ; i < s.size() ; i++){
            if(isPalindrome(s, startindex, i)){//是回文串
//substr是string的成员函数，返回从startindex到i - startindex + 1位置的字符串
                string str = s.substr(startindex , i - startindex + 1);
                path.push_back(str);
            }else{//不是回文串开始回溯
                continue;
            }
            backstr(s , i+1);
            path.pop_back();
        }
    }
    //判断回文串的函数
    bool isPalindrome(const string& s,int start, int end){
        for(int i = start, j = end ; i< j ; i++ , j--){
            if(s[i] != s[j]) return false;
        }
        return true;
    }
    vector<vector<string>> partition(string s) {
        result.clear();
        path.clear();
        backstr(s , 0);
        return result;
    }
};
```

> substr是string的成员函数，返回从startindex到i - startindex + 1位置的字符串

## [93. 复原 IP 地址](https://leetcode.cn/problems/restore-ip-addresses/)

```c++
class Solution {
private:
    vector<string> result;// 记录结果
    // startIndex: 搜索的起始位置，pointNum:添加逗点的数量
    void backtracking(string& s, int startIndex, int pointNum) {
        if (pointNum == 3) { // 逗点数量为3时，分隔结束
            // 判断第四段子字符串是否合法，如果合法就放进result中
            if (isValid(s, startIndex, s.size() - 1)) {
                result.push_back(s);
            }
            return;
        }
        for (int i = startIndex; i < s.size(); i++) {
            if (isValid(s, startIndex, i)) { // 判断 [startIndex,i] 这个区间的子串是否合法
                s.insert(s.begin() + i + 1 , '.');  // 在i的后面插入一个逗点
                pointNum++;
                backtracking(s, i + 2, pointNum);   // 插入逗点之后下一个子串的起始位置为i+2
                pointNum--;                         // 回溯
                s.erase(s.begin() + i + 1);         // 回溯删掉逗点
            } else break; // 不合法，直接结束本层循环
        }
    }
    // 判断字符串s在左闭又闭区间[start, end]所组成的数字是否合法
    bool isValid(const string& s, int start, int end) {
        if (start > end) {
            return false;
        }
        if (s[start] == '0' && start != end) { // 0开头的数字不合法
                return false;
        }
        int num = 0;
        for (int i = start; i <= end; i++) {
            if (s[i] > '9' || s[i] < '0') { // 遇到非数字字符不合法
                return false;
            }
            num = num * 10 + (s[i] - '0');
            if (num > 255) { // 如果大于255了不合法
                return false;
            }
        }
        return true;
    }
public:
    vector<string> restoreIpAddresses(string s) {
        result.clear();
        if (s.size() < 4 || s.size() > 12) return result; // 算是剪枝了
        backtracking(s, 0, 0);
        return result;
    }
};
```

## [78. 子集](https://leetcode.cn/problems/subsets/)

```c++
class Solution {
public:
    vector<vector<int>>result;
    vector<int>path;
    void backnums(vector<int>& nums, int startindex){
        result.push_back(path);
        if(startindex >= nums.size()) return;
        for(int i = startindex ; i < nums.size() ; i++){
            path.push_back(nums[i]);
            backnums(nums , i + 1);
            path.pop_back();
        }
    }
    vector<vector<int>> subsets(vector<int>& nums) {
        result.clear();
        path.clear();
        backnums(nums , 0);
        return result;
    }
};
```

## [90. 子集 II](https://leetcode.cn/problems/subsets-ii/)

```c++
class Solution {
private:
    vector<vector<int>> result;
    vector<int> path;
    void backtracking(vector<int>& nums, int stareindex , vector<bool>& used) {
        result.push_back(path);
        for (int i = stareindex; i < nums.size(); i++) {
            if(i > 0 && nums[i] == nums[i-1] && used[i - 1] == false){
                continue;
            }
            path.push_back(nums[i]);
            used[i] = true;
            backtracking(nums, i + 1 , used);
            used[i] = false;
            path.pop_back();
        }
    }
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        result.clear();
        path.clear();
        vector<bool> used(nums.size(), false);
        sort(nums.begin(),nums.end());
        backtracking(nums, 0, used);
        return result;
    }
};

```

## [491. 非递减子序列](https://leetcode.cn/problems/non-decreasing-subsequences/)

```c++
class Solution {
public:
    vector<int>path;
    vector<vector<int>>result;
    void backnums(vector<int>& nums,int startindex){
        if(path.size() > 1){
            result.push_back(path);
            // 注意这里不要加return，要取树上的节点
        }
        unordered_set<int> uset;
        for (int i = startindex; i < nums.size(); i++) {
            if ((!path.empty() && nums[i] < path.back())
                    || uset.find(nums[i]) != uset.end()) {
                    continue;
            }
            uset.insert(nums[i]); // 记录这个元素在本层用过了，本层后面不能再用了
            path.push_back(nums[i]);
            backnums(nums, i + 1);
            path.pop_back();
        }
    }
    vector<vector<int>> findSubsequences(vector<int>& nums) {
        result.clear();
        path.clear();
        backnums(nums , 0);
        return result;
    }
};
```

## [46. 全排列](https://leetcode.cn/problems/permutations/)

```c++
class Solution {
public:
    vector<int>path;
    vector<vector<int>>result;
    void backnums(vector<int>& nums , vector<bool>& used){
        if(path.size() == nums.size()){
            result.push_back(path);
            return;
        }
        for(int i = 0 ; i< nums.size() ; i++){
            if(used[i] == true) continue;
            used[i] = true;
            path.push_back(nums[i]);
            backnums(nums , used);
            path.pop_back();
            used[i] = false;
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        result.clear();
        path.clear();
        vector<bool> used(nums.size() , false);//定义了used向量，长度与nums相同，元素都被初始化为false,用于标记元素。
        backnums(nums , used);
        return result;
    }
};
```

## [47. 全排列 II](https://leetcode.cn/problems/permutations-ii/)

```c++
class Solution {
public:
    vector<int>path;
    vector<vector<int>>result;
    void backnums(vector<int>& nums , vector<bool>& used){
        if(path.size() == nums.size()){
            result.push_back(path);
            return;
        }
        for(int i = 0 ; i< nums.size() ; i++){
            if(i > 0 && nums[i] == nums[i - 1] && used[i - 1] == false) continue;//去重模板
            if(used[i] == false){
            used[i] = true;
            path.push_back(nums[i]);
            backnums(nums , used);
            path.pop_back();
            used[i] = false;
            }
        }
    }
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        result.clear();
        path.clear();
        sort(nums.begin() , nums.end());
        vector<bool> used(nums.size() , false);//定义了used向量，长度与nums相同，元素都被初始化为false,用于标记元素。
        backnums(nums , used);
        return result;
    }
};
```

> **组合问题和排列问题是在树形结构的叶子节点上收集结果，而子集问题就是取树上所有节点的结果**。
>
> **对于排列问题，树层上去重和树枝上去重，都是可以的，但是树层上去重效率更高！**

## [332. 重新安排行程](https://leetcode.cn/problems/reconstruct-itinerary/)

```c++
class Solution {  
private:  
    // 定义一个unordered_map，其中键是出发机场，值是另一个map。  
    // 外部map的键是到达机场，值是航班次数。  
    unordered_map<string, map<string, int>> targets;  
  
    // 回溯算法，用于查找可能的行程路线  
    bool backtracking(int ticketNum, vector<string>& result) {  
        // 如果已找到所有航班路线，则返回true  
        if (result.size() == ticketNum + 1) {  
            return true;  
        }  
          
        // 遍历当前机场的所有可达机场和对应的航班次数  
        for (pair<const string, int>& target : targets[result[result.size() - 1]]) {  
            // 如果某个可达机场还有剩余航班次数，则尝试选择这个航班  
            if (target.second > 0) {   
                result.push_back(target.first); // 将可达机场添加到结果中  
                target.second--; // 减少该航班的剩余次数  
                  
                // 递归调用backtracking，继续查找下一个航班  
                if (backtracking(ticketNum, result)) return true;  
                  
                // 如果回溯，则将可达机场从结果中移除，并将该航班的剩余次数恢复  
                result.pop_back();  
                target.second++;  
            }  
        }  
        return false; // 如果没有找到可行的航班路线，则返回false  
    }  
  
public:  
    // 主方法，用于查找可能的行程路线  
    vector<string> findItinerary(vector<vector<string>>& tickets) {  
        targets.clear(); // 清除已有的航班信息，为新的输入做准备  
        vector<string> result; // 存储结果的vector  
          
        // 构建出发和到达机场之间的映射关系  
        for (const vector<string>& vec : tickets) {  
            targets[vec[0]][vec[1]]++; // 将ticket添加到targets中，表示从vec[0]到vec[1]的航班次数加1  
        }  
    
        result.push_back("JFK"); // 从JFK机场开始行程，这是一个起始机场  
        backtracking(tickets.size(), result); // 使用回溯算法查找可能的行程路线  
        return result; // 返回找到的行程路线  
    }  
};

```

## [51. N 皇后](https://leetcode.cn/problems/n-queens/)

```c++
class Solution {
private:
vector<vector<string>> result;
// n 为输入的棋盘大小
// row 是当前递归到棋盘的第几行了
void backtracking(int n, int row, vector<string>& chessboard) {
    if (row == n) {
        result.push_back(chessboard);
        return;
    }
    for (int col = 0; col < n; col++) {
        if (isValid(row, col, chessboard, n)) { // 验证合法就可以放
            chessboard[row][col] = 'Q'; // 放置皇后
            backtracking(n, row + 1, chessboard);
            chessboard[row][col] = '.'; // 回溯，撤销皇后
        }
    }
}
bool isValid(int row, int col, vector<string>& chessboard, int n) {
    // 检查列
    for (int i = 0; i < row; i++) { // 这是一个剪枝
        if (chessboard[i][col] == 'Q') {
            return false;
        }
    }
    // 检查 45度角是否有皇后
    for (int i = row - 1, j = col - 1; i >=0 && j >= 0; i--, j--) {
        if (chessboard[i][j] == 'Q') {
            return false;
        }
    }
    // 检查 135度角是否有皇后
    for(int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
        if (chessboard[i][j] == 'Q') {
            return false;
        }
    }
    return true;
}
public:
    vector<vector<string>> solveNQueens(int n) {
        result.clear();
        std::vector<std::string> chessboard(n, std::string(n, '.'));
        backtracking(n, 0, chessboard);
        return result;
    }
};
```

# 贪心算法

## 什么是贪心

**贪心的本质是选择每一阶段的局部最优，从而达到全局最优**。

这么说有点抽象，来举一个例子：

例如，有一堆钞票，你可以拿走十张，如果想达到最大的金额，你要怎么拿？

指定每次拿最大的，最终结果就是拿走最大数额的钱。

每次拿最大的就是局部最优，最后拿走最大数额的钱就是推出全局最优。

## 贪心一般解题步骤

贪心算法一般分为如下四步：

- 将问题分解为若干个子问题
- 找出适合的贪心策略
- 求解每一个子问题的最优解
- 将局部最优解堆叠成全局最优解

## [409. 最长回文串](https://leetcode.cn/problems/longest-palindrome/)

```c++
class Solution {
public:
    int longestPalindrome(string s) {
        unordered_map<char,int>map;
        for(char c : s){//重复出现字符次数
            map[c]++;
        }
        int result = 0 , odd = 0;
        for(const auto& pair : map){
            if(pair.second % 2 == 0){//pair.second代表的map的值，pair.first代表键
                result += pair.second;
            }else{
                odd = max(odd , pair.second);
                if(pair.second > 1) result += (pair.second - 1);//将重复字符数量大于1的取它的偶数次作为回文
            }
        }
        if(odd > 0){//出现次数为奇数，放在回文串中间
            result++;
        }
        return result;
    }
};
//平常要记录重复字符的数量可以使用for(char c : s)
```

`keyCounts.find(key) != keyCounts.end()`这个表达式用于检查`key`是否存在于`keyCounts`映射中。如果`find`返回的迭代器不等于`end`迭代器，那么说明`key`存在于映射中，因此该表达式的结果为`true`；否则，如果`key`不存在，`find`返回`end`迭代器，该表达式的结果为`false`。

## [56. 合并区间](https://leetcode.cn/problems/merge-intervals/)

```c++
class Solution {
public:
    static bool cmp(const vector<int>&a , const vector<int>&b){
        return a[0] < b[0];//维护左边界可以确定左边排序，不用判断左边界大小进行选择
    }
    vector<vector<int>> merge(vector<vector<int>>& nums) {
        if(nums.size() == 0) return nums;
        sort(nums.begin() , nums.end() , cmp);
        vector<vector<int>>result;
        result.push_back(nums[0]);
        for(int i = 1 ; i < nums.size() ; i++){
            if(result.back()[1] >= nums[i][0]){//重叠区间
                result.back()[1] = max(result.back()[1] , nums[i][1]);//更新结果区间右边界最大值
            }else{
                result.push_back(nums[i]);
            }
        }
        return result;
    }
};
//result.back()[1]result中最后一个区间的第二个元素，也就是本题的最后一个元素，result.back()返回最后一个元素
```

## [1953. 你可以工作的最大周数](https://leetcode.cn/problems/maximum-number-of-weeks-for-which-you-can-work/)

```c++
class Solution {
public:
    long long numberOfWeeks(vector<int>& milestones) {
        int mx = *max_element(milestones.begin(), milestones.end());
        long long s = accumulate(milestones.begin(), milestones.end(), 0LL);
        long long rest = s - mx;
        return mx > rest + 1 ? rest * 2 + 1 : s;
    }
};
/*	我们记所有项目的阶段任务数之和为 sss，最大的阶段任务数为 mxmxmx，那么其余所有项目的阶段任务数之和为 rest=s−mxrest = s - mxrest=s−mx。
	如果 mx>rest+1mx \gt rest + 1mx>rest+1，那么就不能完成所有阶段任务，最多只能完成 rest×2+1rest \times 2 + 1rest×2+1 个阶段任务。否则，我们可以完成所有阶段任务，数量为 sss。

```



# 单调栈

**精髓：**及时弹出无用数据，保证栈中数据有序

单调栈中推入的基本都是下标，因为下标可以找到值，题目字眼“下一个更大或更小的元素”，基本为单调栈。

while里写的是符合题目的条件,栈中保存下标还是元素，如果给两份数组对照就保存元素，如果在一个数组中对照就保存下标。

## [739. 每日温度](https://leetcode.cn/problems/daily-temperatures/)

```c++
//从右往左的写法
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        stack<int>st;
        int n = temperatures.size();
        vector<int>ans(n);
        for(int i = n - 1; i >= 0; i--){
            int t = temperatures[i];
            while(!st.empty() && t >= temperatures[st.top()]){
                st.pop();
            }
            if(!st.empty()){
                ans[i] = st.top() - i;
            }
            st.push(i);
        }
        return ans;
    }
};
//从左往右的写法
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        stack<int>st;
        int n = temperatures.size();
        vector<int>ans(n);
        for(int i = 0; i < n; i++){
            int t = temperatures[i];
            while(!st.empty() && t > temperatures[st.top()]){
                int j = st.top();
                st.pop();
                ans[j] = i - j;
            }
            st.push(i);
        }
        return ans;
    }
};
```

## [1475. 商品折扣后的最终价格](https://leetcode.cn/problems/final-prices-with-a-special-discount-in-a-shop/) 1212

```c++
//单调栈写法
class Solution {
public:
    vector<int> finalPrices(vector<int>& prices) {
        int n = prices.size();
        stack<int>st;
        for(int i = 0; i < n; i++){
            int t = prices[i];
            while(!st.empty() && prices[i] <= prices[st.top()]){
                prices[st.top()] -= prices[i]; 
                st.pop();
            }
            st.push(i);
        }
        return prices;
    }
};
//while里写的是符合题目的条件

//暴力写法
class Solution {
public:
    vector<int> finalPrices(vector<int>& prices) {
        int n = prices.size();
        for(int i = 0; i < n - 1; i++){
            for(int j = i + 1; j < n; j++){
                if(prices[j] <= prices[i]){
                    prices[i] -= prices[j];
                    break;
                }
            }
        }
        return prices;
    }
};
```

## [496. 下一个更大元素 I](https://leetcode.cn/problems/next-greater-element-i/)

```c++
//单调栈写法
class Solution {
public:
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        vector<int>ans(nums1.size());
        stack<int>st;
        unordered_map<int,int>map;
        for(int i = 0; i < nums2.size(); i++){
            while(!st.empty() && nums2[i] > st.top()){
                map[st.top()] = nums2[i]; 
                st.pop();
            }
            st.push(nums2[i]);
        }
        while(!st.empty()){
            map[st.top()] = -1;
            st.pop();
        }
        for(int i = 0; i < nums1.size(); i++){
            ans[i] = map[nums1[i]];
        }
        return ans;
    }
};

//暴力写法
class Solution {
public:
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        vector<int>ans(nums1.size());
        for(int i = 0; i < nums1.size(); i++){
            int j = 0;
            while(j < nums2.size() && nums1[i] != nums2[j]) j++;
            while(j < nums2.size() && nums1[i] >= nums2[j]) j++;
            ans[i] = j < nums2.size() ? nums2[j] : -1;
        }
        return ans;
    }
};
```

## [503. 下一个更大元素 II](https://leetcode.cn/problems/next-greater-element-ii/)

```c++
class Solution {
public:
    vector<int> nextGreaterElements(vector<int>& nums) {
        int n = nums.size();
        vector<int>ans(n, -1);
        stack<int>st;
        for(int i = 0; i < 2 * n - 1; i++){
            while(!st.empty() && nums[st.top()] < nums[i % n]){
                ans[st.top()] = nums[i % n];
                st.pop();
            }
            st.push(i % n);
        }
        return ans;
    }
};
/*i < 2 * n - 1，就是将数组遍历两遍，最后一个元素不用遍历
为何最后一个元素不用遍历呢？因为第一次遍历的时候最后一个元素已经和前面的元素都比较过了，第二次遍历的时候就没有必要了
```

## [1019. 链表中的下一个更大节点](https://leetcode.cn/problems/next-greater-node-in-linked-list/) 1571

```c++
class Solution {
public:
    vector<int> nextLargerNodes(ListNode* head) {
        vector<int>ans;
        stack<pair<int,int>>st;//first:val；second：下标
        for(auto cur = head; cur; cur = cur->next){
            while(!st.empty() && st.top().first < cur->val){
                ans[st.top().second] = cur->val;
                st.pop();
            }
            st.emplace(cur->val, ans.size());
            ans.push_back(0);

        }
        return ans;
    }
};
/*这里的pair由程序员自定义，emplace构造了一个对象在栈中，cur->val表示为first，ans.size()表示为潜在的下标，代表second*/

//使用make_pair也行
class Solution {
public:
    vector<int> nextLargerNodes(ListNode* head) {
        vector<int>ans;
        stack<pair<int,int>>st;//first:val；second：下标
        int index = 0;//下标计数
        for(auto cur = head; cur; cur = cur->next){
            while(!st.empty() && st.top().first < cur->val){
                ans[st.top().second] = cur->val;
                st.pop();
            }
            st.push(make_pair(cur->val, index++));
            ans.push_back(0);
        }
        return ans;
    }
};
```

## [962. 最大宽度坡](https://leetcode.cn/problems/maximum-width-ramp/) 1608

```c++
class Solution{
public:
    int maxWidthRamp(vector<int> &nums){
        stack<int> st;
        int res = 0;
        int n = nums.size();
        for (int i = 0; i < n; ++i){
            if (st.empty() || nums[st.top()] > nums[i]) st.push(i);
        }
        for (int j = n - 1; j >= res; --j){ 
            while (st.size() && nums[st.top()] <= nums[j]){
                int pos = st.top();
                st.pop();
                res = max(res, j - pos);
            }
        }
        return res;
    }
};
/*此题的单调栈跟其他题有个不同的点就是，普通单调栈会不断刷新，弹出不符合要求的元素，此题不会
```

## [907. 子数组的最小值之和](https://leetcode.cn/problems/sum-of-subarray-minimums/) 1976

```c++
class Solution {
public:
    const int mod = 1e9 + 7;
    int sumSubarrayMins(vector<int>& arr) {
        long ans = 0l;
        arr.push_back(-1);
        stack<int>st;
        st.push(-1);
        for(int r = 0; r < arr.size(); r++){
            while(st.size() > 1 && arr[st.top()] >= arr[r]){
                int i = st.top();
                st.pop();
                ans += (long)arr[i] * (i - st.top()) * (r - i);
            }
            st.push(r);
        }
        return ans % mod;
    }
};
```

## [84. 柱状图中最大的矩形](https://leetcode.cn/problems/largest-rectangle-in-histogram/)

```c++
class Solution {
public:
    int largestRectangleArea(vector<int>& heights){
        int ans = 0;
        vector<int> st;
        heights.insert(heights.begin(), 0);
        heights.push_back(0);
        for (int i = 0; i < heights.size(); i++){
            while (!st.empty() && heights[st.back()] > heights[i]){
                int cur = st.back();
                st.pop_back();
                int left = st.back() + 1;
                int right = i - 1;
                ans = max(ans, (right - left + 1) * heights[cur]);
            }
            st.push_back(i);
        }
        return ans;
    }
};
```

## [LCR 040. 最大矩形](https://leetcode.cn/problems/PLYXKQ/)

```c++
class Solution {
public:
    int maximalRectangle(vector<string>& matrix) {
        if(matrix.size() == 0) return 0;
        int n = matrix.size(), m = matrix[0].length();
        // 把每一行看出一个柱形图的底部，记录柱形图高度
        vector<vector<int>> heights(n, vector<int>(m+1, 0)); 
        for(int i=0; i<n; ++i){
            for(int j=0; j<m; ++j){
                if(i == 0) heights[0][j] = matrix[0][j] - '0';
                else heights[i][j] = (matrix[i][j] == '0')? 0 : heights[i-1][j] + 1;
            }
        }
        // 逐行用单调栈计算最大矩形
        int maxarea = 0;
        for(int i=0; i<n; ++i){
            stack<int> st;
            st.push(-1);
            for(int j=0; j<m+1; ++j){
                while(st.top()!=-1 && heights[i][st.top()] > heights[i][j]){
                    int idx = st.top();
                    st.pop();
                    maxarea = max(maxarea, heights[i][idx] * (j - 1 - st.top()));
                }
                st.push(j);
            }
        }
        return maxarea;
    }
};
```

## [316. 去除重复字母](https://leetcode.cn/problems/remove-duplicate-letters/)

```c++
//此题要使用记录次数的方法统计
class Solution {
public:
    string removeDuplicateLetters(string s) {
        vector<int> map(26, 0); // 记录字符串元素出现个数(为了保住最后一个元素不被弹出)
        vector<bool> visited(26, false); // 入栈登记表
        string result; // 栈本体 这里可以直接用string作为栈体
        for (char c : s) map[c - 'a']++; // 准备入栈各元素数量登记
        for (char c : s) {
            // 如果该元素进去过了 就不给进了 别忘了进不去的元素也要把数量减一
            if (visited[c - 'a']) { 
                map[c - 'a']--;
                continue;
            }
            // 如果栈不空 栈顶大于准备压入的元素 栈顶元素不是最后一个幸存者 就要把栈顶弹出来
            while (!result.empty() && result.back() > c && map[result.back() - 'a'] > 0) {
                visited[result.back() - 'a'] = false; // 做好访客登记
                result.pop_back();               
            }
            result.push_back(c); // 新客入栈
            visited[c - 'a'] = true;       
            map[c - 'a']--;
        }
        return result;
    }
};
```

## [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/) 

```c++
class Solution {
public:
//分为三根柱子，分别为前中后,由此组成了一个桶
//height[i]为后，mid_h为中,left为前
    int trap(vector<int>& height) {
        stack<int>st;
        int n = height.size();
        int ans = 0;
        for(int i = 0; i < n; i++){
            //大压小
            while(!st.empty() && height[i] >= height[st.top()]){
                int mid_h = height[st.top()];//如[1，2，3]，mid_h保存的就是3
                st.pop();//然后弹出3
                if(st.empty()) break;//如果栈为空，说明接不了雨水
                int left = st.top();//现还剩[1,2]，left保存的就是2
                int h = min(height[left], height[i]) - mid_h;//桶的高度就是前后的最小高度减去中间柱子的高度
                ans += h * (i - left - 1);//桶的宽就是后减去前
            }
            st.push(i);
        }
        return ans;
    }
};
```

## [402. 移掉 K 位数字](https://leetcode.cn/problems/remove-k-digits/) ~1800

```c++
class Solution {
public:
    string removeKdigits(string num, int k) {
        stack<char>st;
        int n = num.size();
        for(int i = 0; i < n; i++){
            while(k && !st.empty() && st.top() > num[i]){
                st.pop();
                k--;
            }
            if(st.empty() && num[i] == '0') continue;
            st.push(num[i]);
        }
        string res;
        while(!st.empty()){
            if(k > 0) k--;
            else if(k == 0) res += st.top();
            st.pop();
        }
        reverse(res.begin(), res.end());
        return res == "" ? "0" : res;
    }
};
```

## [1673. 找出最具竞争力的子序列](https://leetcode.cn/problems/find-the-most-competitive-subsequence/) 1802

```c++
```



# 位运算

(a + b - 1) / b可以进行向上取整。

非负数向右移一位等于除2

```c++
l + ((r - l) >> 1);//二分算法求中点常用
```

```c++
>>4//是将整个二进制右移4位
<<4//左移类似
```

* 1、判断一个整数是否是2的幂
* 2、判断一个整数是否是3的幂
* 3、已知n是非负数，返回大于等于n的最小的2某次方
* 4、已知区间[left， right]内所有数字 & 的结果
* 5、反转一个二进制的状态，不是0变1、1变0，是逆序
* 6、返回一个数二进制中有几个1



集合{0,2,3}可以压缩成2^0^+2^2^+2^3^=13,也就是二进制数 （1101）~2~。

利用位运算「并行计算」的特点，我们可以高效地做一些和集合有关的运算。按照常见的应用场景，可以分为以下四类：

1. 集合与集合
2. 集合与元素
3. 遍历集合
4. 枚举集合

## 遍历集合

设元素范围从 0到 𝑛−1，挨个判断每个元素是否在集合 𝑠中：

```c++
for (int i = 0; i < n; i++) {
    if ((s >> i) & 1) { // i 在 s 中
        // 处理 i 的逻辑
    }
}
```

## 枚举集合

设元素范围从 0 到 𝑛−1，从空集 ∅∅ 枚举到全集 U：

```c++
for (int s = 0; s < (1 << n); s++) {
    // 处理 s 的逻辑
}
```

## 枚举非空子集

设集合为 𝑠*s*，**从大到小**枚举 𝑠 的所有**非空**子集 sub：

```c++
for (int sub = s; sub; sub = (sub - 1) & s) {
    // 处理 sub 的逻辑
}
```

## 枚举子集（包含空集）

如果要从大到小枚举 𝑠的所有子集 sub（从 𝑠 枚举到空集 ∅），可以这样写：

```c++
int sub = s;
do {
    // 处理 sub 的逻辑
    sub = (sub - 1) & s;
} while (sub != s);
```

## [231. 2 的幂](https://leetcode.cn/problems/power-of-two/)

判断一个数是否是2的幂：

```c++
//2的幂的二进制只有一个1
//2： 0010
//4： 0100
//8： 1000
//n-1后，1变0,0变1
return n > 0 && (n & n (n - 1)) == 0;
```



```c++
class Solution {
public:
    bool isPowerOfTwo(int n) {
        return n > 0 && n == (n & -n);
    }
};
```

## [326. 3 的幂](https://leetcode.cn/problems/power-of-three/)

```c++
class Solution {
public:
    bool isPowerOfThree(int n) {
        return n > 0 && 1162261467 % n == 0;//int范围内最大的3次幂，3^19=1162261467
    }
};
```

## [342. 4的幂](https://leetcode.cn/problems/power-of-four/)

```c++
class Solution {
public:
    bool isPowerOfFour(int n) {
        return n > 0 && (n & -n) == n && n % 3 == 1;
    }
};
//4的幂mod3必余1
```

## [201. 数字范围按位与](https://leetcode.cn/problems/bitwise-and-of-numbers-range/) 题型4

```c++
class Solution {
public:
    int rangeBitwiseAnd(int left, int right) {
        while(left < right){
            right -= right & -right;//right不断减去最右侧的1，直到left>=right
        }
        return right;
    }
};
```

## 异或运算

**常见题型：**

* 1、交换两个数
* 2、代替判断语句（偏底层）
* 3、找到缺失的数字（**性质：**一个完整数组的异或和异或上缺失数字数组的异或和即可得到缺失的那个数字）
* 4、数组中一种数出现了奇数次，其它数出现了偶数次，返回奇数次的数
* 5、数组中有两种数出现了奇数次，其它数出现了偶数次，返回这两种出现了奇数次的数
* 6、数组中只有一种数出现次数少于m次，其它数都出现了m次，返回出现次数小于m次的那种数

### [136. 只出现一次的数字](https://leetcode.cn/problems/single-number/) 题型4

```c++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int eor = 0;
        for(int x : nums){
            eor ^= x;
        }
        return eor;
    }
};
```

## Brian Kernighan算法

用于题型5、6！！！

**取最右侧的1**

```c++
n =  01101000
过程：~n+1（变为补码），再与n相与就可得到最右侧的1，即n &(~n + 1)或n & (-n)
答案：00001000
```

### [260. 只出现一次的数字 III](https://leetcode.cn/problems/single-number-iii/)

```c++
class Solution {
public:
    vector<int> singleNumber(vector<int>& nums) {
        long long eor1 = 0;//int取反直接爆溢出
        for(auto &x : nums){
            eor1 ^= x;
        }
        //取最右侧的1
        int rightOne = eor1 & (-eor1);
        int eor2 = 0;
        for(auto &y : nums){
            if((y & rightOne) == 0){
                eor2 ^= y;
            }
        }
        int a = eor2;
        int b =  eor1 ^ eor2;
        return {a, b};
    }
};
```

### [137. 只出现一次的数字 II](https://leetcode.cn/problems/single-number-ii/)

```c++
class Solution {
public:
    int singleNumber(vector<int> &nums) {
        int ans = 0;
        for (int i = 0; i < 32; i++) {//设置为32位是因为整数是32位宽
            int cnt1 = 0;
            for (int x: nums) {
                cnt1 += x >> i & 1;//将所有元素的比特位想加
            }
            ans |= cnt1 % 3 << i;//还原只出现一次的那个数字的比特位
        }
        return ans;
    }
};
```



# 递归



```c++
```



# 动态规划

## 理论基础

​	动态规划，英文：Dynamic Programming，简称DP，如果某一问题有很多重叠子问题，使用动态规划是最有效的。

​	所以动态规划中每一个状态一定是由上一个状态推导出来的，**这一点就区分于贪心**，贪心没有状态推导，而是从局部直接选最优的。

​	动态转移方程就是尝试策略！！！由小问题推出大问题。

​	找到**可变参数**，确定动态规划dp的长度，动态规划都是由暴力尝试->记忆化搜索->动态规划的改善，记忆化搜索其实就是暴力尝试挂上缓存，而动态规划的dp表就是存递归的数值，返回已经计算过的数值。

**动态规划五步曲：**

1. 确定dp数组（dp table）以及下标的含义（下标即可变参数）
2. 确定递推公式
3. dp数组如何初始化
4. 确定遍历顺序
5. 举例推导dp数组

## [509. 斐波那契数](https://leetcode.cn/problems/fibonacci-number/)

```c++
//方法1、维护整个数列
class Solution {
public:
    int fib(int n) {
        vector<int>dp(n + 1);
        dp[0] = 0;
        dp[1] = 1;
        int res = 0;
        for(int i = 2; i <= n; i++){
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
};
//为什么 i <= n 并且dp数组长度为n + 1?
//前两个数被初始化为0和1了,如果 n 是3，那么循环将计算 dp[2] 和 dp[3] 的值。如果 n 是4，那么将计算 dp[2]、dp[3] 和 dp[4] 的值，以此类推。所以循环条件是 i <= n 是正确的。
//因为数组是从0开始索引的，所以我们需要一个长度为 n + 1 的数组来存储从 dp[0] 到 dp[n] 的值。

//方法2、维护两个元素
class Solution {
public:
    int fib(int n) {
        if(n < 2) return n;
        vector<int>dp(2);
        dp[0] = 0;
        dp[1] = 1;
        for(int i = 2; i <= n; i++){
            int sum = dp[0] + dp[1];
            dp[0] = dp[1];
            dp[1] = sum;
        }
        return dp[1];
    }
};

//方法3、递归
class Solution {
public:
    int fib(int n) {
        if(n < 2) return n;
        return fib(n - 1) + fib(n - 2);
    }
};
```

## [70. 爬楼梯](https://leetcode.cn/problems/climbing-stairs/)

```c++
class Solution {
public:
    int climbStairs(int n) {
        if (n <= 1) return n; // 因为下面直接对dp[2]操作了，防止空指针
        vector<int> dp(n + 1);
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) { // 注意i是从3开始的
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
};
```

## [62. 不同路径](https://leetcode.cn/problems/unique-paths/)

```c++
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>>dp(m,vector<int>(n, 0));//二维数组，m行n列，元素初始化为0
        for(int i = 0; i < m; i++) dp[i][0] = 1;
        for(int j = 0; j < n; j++) dp[0][j] = 1;
        for(int i = 1; i < m; i++){
            for(int j = 1; j < n; j++){
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }
};
```

## [63. 不同路径 II](https://leetcode.cn/problems/unique-paths-ii/)

```c++
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size(), n = obstacleGrid[0].size();
        vector<vector<int>>dp(m, vector<int>(n, 0));
        if(obstacleGrid[m - 1][n - 1] == 1 || obstacleGrid[0][0] == 1) return 0;
        for(int i = 0; i < m && obstacleGrid[i][0] == 0; i++) dp[i][0] = 1;
        for(int j = 0; j < n && obstacleGrid[0][j] == 0; j++) dp[0][j] = 1;
        for(int i = 1; i < m; i++){
            for(int j = 1; j < n; j++){
                if(obstacleGrid[i][j] == 0){
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        return dp[m - 1][n - 1];
    }
};
```

## [377. 组合总和 Ⅳ](https://leetcode.cn/problems/combination-sum-iv/)（记忆性回溯；动态规划）

```c++
class Solution {
public:
    int combinationSum4(vector<int> &nums, int target) {
        vector<unsigned> f(target + 1);
        f[0] = 1;
        for (int i = 1; i <= target; i++) {
            for (int x : nums) {
                if (x <= i) {
                    f[i] += f[i - x];
                }
            }
        }
        return f[target];
    }
};
```

## [746. 使用最小花费爬楼梯](https://leetcode.cn/problems/min-cost-climbing-stairs/)

```c++
class Solution {
public:
    int minCostClimbingStairs(vector<int>& cost) {
        vector<int>dp(cost.size() + 1);
        dp[0] = 0;
        dp[1] = 0;
        for(int i = 2; i <= cost.size(); i++){
            dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]);
        }
        return dp[cost.size()];
    }
};
//此题的可变参数为cost[i]即花费
```

## [LCR 168. 丑数](https://leetcode.cn/problems/chou-shu-lcof/)

```c++
class Solution {
public:
    int nthUglyNumber(int n) {
        vector<int>dp(n + 1);
        dp[1] = 1;
        for(int i = 2, i2 = 1, i3 = 1, i5 = 1, a, b, c; i <= n; i++){
            a = dp[i2] * 2;
            b = dp[i3] * 3;
            c = dp[i5] * 5;
            int cur = min(min(a, b), c);
            if(cur == a) i2++;
            if(cur == b) i3++;
            if(cur == c) i5++;
            dp[i] = cur;
        }
        return dp[n];
    }
};
//定义3个指针，根据cur的值更新相应的指针。如果cur等于由2生成的丑数a，则i2指针向后移动一位；如果cur等于由3生成的丑数b，则i3指针向后移动一位；如果cur等于由5生成的丑数c，则i5指针向后移动一位。
```

## [32. 最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/)

```c++
class Solution {
public:
    int longestValidParentheses(string s) {
        int n = s.size();
        vector<int>dp(n);
        int ans = 0, pre;
        for(int i = 1; i < n; i++){
            if(s[i] == ')'){
                pre = i - dp[i - 1] - 1;
                if(pre >= 0 && s[pre] == '('){
                    dp[i] = dp[i - 1] + 2 + (pre > 0 ? dp[pre - 1] : 0);
                }
            }
            ans = max(ans, dp[i]);
        }
        return ans;
    }
};
```

## [90. 子集 II](https://leetcode.cn/problems/subsets-ii/)

```c++
class Solution {
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        vector<vector<int>> result;
        vector<int> subset;
        sort(nums.begin(), nums.end()); // 首先对数组进行排序，以便于去重
        backtrack(nums, 0, subset, result);
        return result;
    }

private:
    void backtrack(const vector<int>& nums, int start, vector<int>& subset, vector<vector<int>>& result) {
        result.push_back(subset); // 将当前子集添加到结果中

        for (int i = start; i < nums.size(); ++i) {
            // 如果当前元素与前一个元素相同，则跳过以避免重复的子集
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            subset.push_back(nums[i]); // 做选择
            backtrack(nums, i + 1, subset, result); // 递归生成子集
            subset.pop_back(); // 撤销选择
        }
    }
};
```

## [LCR 083. 全排列](https://leetcode.cn/problems/VvJkup/)

```c++
class Solution {
public:
    vector<vector<int>> permute(vector<int> &nums) {
        int n = nums.size();
        vector<vector<int>> ans;
        vector<int> path(n), on_path(n);
        function<void(int)> dfs = [&](int i) {
            if (i == n) {
                ans.emplace_back(path);
                return;
            }
            for (int j = 0; j < n; ++j) {
                if (!on_path[j]) {
                    path[i] = nums[j];
                    on_path[j] = true;
                    dfs(i + 1);
                    on_path[j] = false; // 恢复现场
                }
            }
        };
        dfs(0);
        return ans;
    }
};
```



# 图论





# 每日一题

## [1026. 节点与其祖先之间的最大差值](https://leetcode.cn/problems/maximum-difference-between-node-and-ancestor/)

```c++
class Solution {  
public:  
    int result = 0; // 存储最终结果的变量，初始化为0  
  
    // 深度优先搜索函数，用于遍历二叉树  
    void dfs(TreeNode *node, int mn, int mx) {  
        if (node == nullptr) return; // 如果节点为空，则直接返回，结束递归  
  
        mn = min(mn, node->val); // 更新当前路径上的最小值  
        mx = max(mx, node->val); // 更新当前路径上的最大值  
  
        // 更新结果，计算当前节点值与最小值的差和最大值与当前节点值的差中的较大值  
        result = max(result, max(node->val - mn, mx - node->val));  
  
        // 递归遍历左子树，传入更新后的最小值和最大值  
        dfs(node->left, mn, mx);  
  
        // 递归遍历右子树，传入更新后的最小值和最大值  
        dfs(node->right, mn, mx);  
    }  
  
    // 求解二叉树中每个节点与其祖先节点值的最大差值  
    int maxAncestorDiff(TreeNode *root) {  
        if (root == nullptr) return 0; // 如果根节点为空，则返回0  
  
        // 调用深度优先搜索函数，从根节点开始遍历，初始的最小值和最大值都设为根节点的值  
        dfs(root, root->val, root->val);  
  
        // 返回最终的结果  
        return result;  
    }  
};
//所谓递归要传入值进去，采用了自顶向下方法，搜索同一路径最大值和最小值，更新最大差值。
//也可使用自顶向下，根据左右子树得到整棵树的情况，类似dp
```

## [738. 单调递增的数字](https://leetcode.cn/problems/monotone-increasing-digits/)(判断)

暴力解法：

```c++
class Solution {
private:
    // 判断一个数字的各位上是否是递增
    bool checkNum(int num) {
        int max = 10;
        while (num) {
            int t = num % 10;
            if (max >= t) max = t;
            else return false;
            num = num / 10;
        }
        return true;
    }
public:
    int monotoneIncreasingDigits(int N) {
        for (int i = N; i > 0; i--) { // 从大到小遍历
            if (checkNum(i)) return i;
        }
        return 0;
    }
};
```



```c++
class Solution {
public:
    int monotoneIncreasingDigits(int N) {
        string strNum = to_string(N);
        // flag用来标记赋值9从哪里开始
        // 设置为这个默认值，为了防止第二个for循环在flag没有被赋值的情况下执行
        int flag = strNum.size();
        for (int i = strNum.size() - 1; i > 0; i--) {
            if (strNum[i - 1] > strNum[i] ) {
                flag = i;
                strNum[i - 1]--;
            }
        }
        for (int i = flag; i < strNum.size(); i++) {
            strNum[i] = '9';
        }
        return stoi(strNum);
    }
};
//stoi可以将字符串数组中的数字，转换为整数
//to_string转换为字符串类型
```

## [1702. 修改后的最大二进制字符串](https://leetcode.cn/problems/maximum-binary-string-after-change/)（判断）

```c++
class Solution {
public:
    string maximumBinaryString(string s) {
        int n = s.size();
        int right_one = 0;
        bool flag = false;//用来标记0的位置
        for(auto c : s){
            if(c == '0'){
                flag = true;
            }else if(flag && c == '1'){//将第一个0后面的1的个数相加
                right_one++;
            }
        }
        if(!flag){//如果全为1
            return s;
        }
        string res(n, '1');//创造一个n长字符串，默认值为全1
        res[n - right_one - 1] = '0';//在最后的答案指定的位置标为0
        return res;
    }
};
//用于寻找位置的例题，要多学会使用标记位置的思想
```

## [2923. 找到冠军 I](https://leetcode.cn/problems/find-champion-i/)

```c++
class Solution {
public:
    int findChampion(vector<vector<int>>& grid) {
        int n = grid.size();
        for(int i = 0; i < n; i++){
            int sum = 0;
            for(auto x : grid[i]) sum += x;
            if(sum == n - 1) return i;
        }
        return -1;
    }
};
//利用了题目给出的特性
//其中矩阵(3*3)示例：
/*grid = [
	[0,0,1],
    [1,0,1],
    [0,0,0]
]*/
//n = 3，if(sum == n - 1) return i;
//n - 1代表的是那个数组中参加比赛的所有队伍（不包括自己），在一个数组中必定分出一个冠军，若没有冠军返回-1，sum代表数组中的参赛队伍，如果符合要求说明那个队伍必定会分出冠军队伍。
```

## 好数（判断）

【题目】

一个整数如果按从低位到高位的顺序，奇数位(个位、百位、万位 · · · )上 的数字是奇数，偶数位(十位、千位、十万位 · · · )上的数字是偶数，我们就称 之为“好数”。
给定一个正整数 N，请计算从 1 到 N 一共有多少个好数。 【输入格式】
一个整数 N。 【输出格式】
一个整数代表答案。
【样例输入 1】 24
			【输出】 ： 7

【样例说明】
24 以内的好数有 1、3、5、7、9、21、23，一共 7 个。

【样例输入 2】 2024
			【输出 】：150

```c++
#include <stdbool.h>
#include <iostream>

using namespace std;

int main(int N){
    cin >> N;
    int res = 0;//结果
    for(int i = 1; i <= N; i++){//遍历每个数
        int top = 1;//用来判断奇偶
        bool flash = true;//用来标记
        int pre = i;//使用另一个变量保存各个位置
        while(pre){
            if(pre % 2 != top){
                flash = false;
                break;//不符合条件跳出循环
            }
            top ^= 1;//异或运算
            pre /= 10;
        }
        if(flash){
            res++;
        }
    }
    cout << "1到N之间共有" << res << "个好数" << endl;
    return 0;
};

/*将异或运算取反即可得到同或运算
bool xnor(bool a, bool b) {  
    return !(a ^ b); // 先进行异或运算，然后对结果进行取反，得到同或运算的结果  
}*/
```

## [2007. 从双倍数组中还原原数组](https://leetcode.cn/problems/find-original-array-from-doubled-array/)（排序+队列）

```c++
class Solution {
public:
    vector<int> findOriginalArray(vector<int>& changed) {
        ranges::sort(changed);//排序
        vector<int>res;//存放结果集
        queue<int>que;//创建一个队列存放双倍元素
        for(int x : changed){
            if(!que.empty()){
                if(que.front() < x){//如果队列顶部元素小于原数组第一个，说明不符合题目要求
                    return {};
                }else if(que.front() == x){//配对成功，删除队列第一个元素
                    que.pop();
                    continue;
                }
            }
            res.push_back(x);
            que.push(x * 2);
        }
        return que.empty() ? res : vector<int>();
    }
};
```

## [1329. 将矩阵按对角线排序](https://leetcode.cn/problems/sort-the-matrix-diagonally/)

````c++
class Solution {  
public:
    vector<vector<int>> diagonalSort(vector<vector<int>>& mat) {
        int n = mat.size(), m = mat[0].size();  
        // 使用哈希表存储每条对角线上的元素  
        // 哈希表的键是对角线的编号（行号 - 列号），值是该对角线上元素的向量，添加一个键为1的条目，其值是一个包含两个整数的向量 如：vs[1] = {2, 3}; 
    	//与unordered_map<int,int>vs2的区别：vs2[1] = 2;    
        unordered_map<int, vector<int>> vs;  
        // 遍历矩阵的每个元素  
        for (int i = 0; i < n; ++i) {  
            for (int j = 0; j < m; ++j) {  
                // 将元素添加到对应对角线的向量中  
                vs[i - j].emplace_back(mat[i][j]);  
            }  
        } 
        for (auto& v : vs) sort(v.second.rbegin(), v.second.rend());  
        // 将排序后的元素放回原矩阵中  
        for (int i = 0; i < n; ++i) {  
            for (int j = 0; j < m; ++j) {  
                // 取出当前对角线向量中的最后一个元素（即排序后的最大元素）  
                mat[i][j] = vs[i - j].back();  
                // 移除已取出的元素  
                vs[i - j].pop_back();  
            }  
        }  
        return mat;  
    } 
};
//左对角线元素的坐标i - j相等，右对角线的元素i + j相等
//对于此题以同一斜边为键，存储斜边上的元素向量，将它们放入一个数组并进行排序再放回去
````

## [2079. 给植物浇水](https://leetcode.cn/problems/watering-plants/) (模拟)**1321**

```c++
class Solution {
public:
    int wateringPlants(vector<int>& plants, int capacity) {
        int n = plants.size(), res = 0;
        int tmp = capacity;
        for(int i = 0; i < n; i++){
            if(capacity >= plants[i]){
                capacity -= plants[i];
                res++;
            }else{
                capacity = tmp - plants[i];
                res += 2 * i + 1;//往返步数            
            }
        }
        return res;
    }
};
```

## [2105. 给植物浇水 II](https://leetcode.cn/problems/watering-plants-ii/) (相向双指针)**1507**

```c++
class Solution {
public:
    int minimumRefill(vector<int>& plants, int capacityA, int capacityB) {
        int n = plants.size(), i = 0, j = n - 1, ans = 0;
        int sA = capacityA, sB = capacityB;
        while (i < j){
            if(plants[i] > sA){
                sA = capacityA;
                ans++;  
            }
            sA -= plants[i++];
            if(plants[j] > sB){
                sB = capacityB;
                ans++; 
            }
            sB -= plants[j--];
        }
        if(i == j && max(sA, sB) < plants[i]){
            ans++;
        }
        return ans;
    }
};
```



# 周赛题单

## 判断

### [3115. 质数的最大距离](https://leetcode.cn/problems/maximum-prime-difference/)

**判断素数板子**

这里的 `i * i <= n` 是一个优化，因为如果 `n` 有一个大于它平方根的因子，那么它必然也有一个小于或等于它平方根的因子。

```c++
    bool isprime(int n) {//判断是否是素数的一个板子：：试除法
        for (int i = 2; i * i <= n; i++) {
            if (n % i == 0) {
                return false;
            }
        }
        return n >= 2;
    }
```

#### 完整代码:

```c++
class Solution {
    bool isprime(int n) {
        for (int i = 2; i * i <= n; i++) {
            if (n % i == 0) {
                return false;
            }
        }
        return n >= 2;
    }

public:
    int maximumPrimeDifference(vector<int>& nums) {
        int i = 0;
        while (!isprime(nums[i])) {
            i++;
        }
        int j = nums.size() - 1;
        while (!isprime(nums[j])) {
            j--;
        }
        return j - i;
    }
};

```

### [3132. 找出与数组相加的整数 II](https://leetcode.cn/problems/find-the-integer-added-to-array-ii/)

​	与[392. 判断子序列](https://leetcode.cn/problems/is-subsequence/)做法一样

```c++
class Solution {
public:
    int minimumAddedInteger(vector<int>& nums1, vector<int>& nums2) {
        ranges::sort(nums1);
        ranges::sort(nums2);
        for(int i = 2; ; i--){
            int cnt = nums2[0] - nums1[i];
            int j = 0;
            for(int k = i; k < nums1.size(); k++){
                if(j < nums2.size() && nums2[j] - nums1[k] == cnt && ++j == nums2.size()) return cnt;
            }
        }
    }
};
//尤其是if中的判断是环环相扣的，前面为真才会执行后面的语句
//392. 判断子序列
class Solution {
public:
    bool isSubsequence(string s, string t) {
        if(s.size() == 0) return true;
        for(int i = 0, j = 0; j < t.size(); j++){
            if(s[i] == t[j]){
                if(++i == s.size()) return true;
            }
        }
        return false;
    }
};
```



## 排序

### [3107. 使数组中位数等于 K 的最少操作数](https://leetcode.cn/problems/minimum-operations-to-make-median-of-array-equal-to-k/)

```c++
class Solution {
public:
    long long minOperationsToMakeMedianK(vector<int>& nums, int k) {
        sort(nums.begin(), nums.end());//先排序找出中位数
        int result = 0;
        int n = nums.size();
        int m = n / 2;
        if(nums.empty()) return 0;
        if(nums[m] > k){//如果中位数大于k，只需修改中位值左边大于k的数字
            for(int i = m; i >= 0 && nums[i] > k; i--){
                result += nums[i] - k;
            }
        }else{//反之相反
            for(int i = m; i < n && nums[i] < k; i++){
                result += k - nums[i];
            }
        }
        return result;
    }
};
```

## 枚举

### [3105. 最长的严格递增或递减子数组](https://leetcode.cn/problems/longest-strictly-increasing-or-strictly-decreasing-subarray/)

```c++
class Solution {
public:
    int longestMonotonicSubarray(vector<int>& nums) {
        int max_length = 1;//最长子数组长度
        int dijian = 1;
        int dizeng = 1;
        if(nums.empty()) return 0;
        for(int i = 1; i < nums.size(); i++){
            if(nums[i - 1] > nums[i]){
                dijian += 1;
                dizeng = 1;
            }else if(nums[i - 1] < nums[i]){
                dizeng +=1;
                dijian = 1;
            }else{
                dijian = 1;
                dizeng = 1;
            }
            max_length = max(max_length , dizeng);
            max_length = max(max_length , dijian);
        }
        return max_length;
    }
};
```

### [3127. 构造相同颜色的正方形](https://leetcode.cn/problems/make-a-square-with-the-same-color/)

```c++
class Solution {
public:
    bool canMakeSquare(vector<vector<char>>& grid) {
        int n=grid.size(),m=grid[0].size(),sum=0;
        
        for(int i=0;i<n-1;i++)
            for(int j=0;j<m-1;j++)
            {
                   sum=(grid[i][j]=='W'?1:0)+(grid[i+1][j]=='W'?1:0)+(grid[i][j+1]=='W'?1:0)+(grid[i+1][j+1]=='W'?1:0);
                   if(!sum || sum==4 || sum==1 || sum==3)  //符合题目要求的四种情况
                       return true;
            }
        return false;
    }
};
//只需统计正方形中一种颜色的数量
```

### [3128. 直角三角形](https://leetcode.cn/problems/right-triangles/)

```c++
class Solution {  
public:  
    long long numberOfRightTriangles(vector<vector<int>>& grid) {  
        int m = grid.size();  
        int n = grid[0].size();  
        long long count = 0;  
        vector<int> rowSums(m, 0);  
        vector<int> colSums(n, 0);  
        for (int i = 0; i < m; ++i) {//利用两个数组遍历每行每列求出1的总数  
            for (int j = 0; j < n; ++j) {  
                rowSums[i] += grid[i][j];  
                colSums[j] += grid[i][j];  
            }  
        }
        for (int i = 0; i < m; ++i) {//找到那个1之后，就相乘就行。  
            for (int j = 0; j < n; ++j) {  
                if (grid[i][j] == 1) { 
                    int sameRow = rowSums[i] - 1; 
                    int sameCol = colSums[j] - 1; 
                    count += (long long)sameRow * sameCol;  
                }  
            }  
        }  
          
        return count;  
    }  
};
//想法是先找到一个1，以它为中心遍历那一行和那一列统计所有1，相乘即可得到答案
```



## 滑动窗口

### [2958. 最多 K 个重复元素的最长子数组](https://leetcode.cn/problems/length-of-longest-subarray-with-at-most-k-frequency/)

```c++
class Solution {
public:
    int maxSubarrayLength(vector<int>& nums, int k) {
        unordered_map<int,int>map;//统计元素频率
        int n = nums.size();
        int res = 0, left = 0;
        for(int right = 0; right < n; right++){
            map[nums[right]]++;//统计右指针元素的频率
            while(map[nums[right]] > k){
                map[nums[left++]]--;//将left所指元素频率减少，并右移left指针
            }
            res = max(res, right - left + 1);//更新窗口内的数组长度
        }
        return res;
    }
};
//数组问题计数多半都是滑动窗口，学会处理好滑动窗口的边界问题很重要
```

### [2009. 使数组连续的最少操作数](https://leetcode.cn/problems/minimum-number-of-operations-to-make-array-continuous/)

```c++
class Solution {
public:
    int minOperations(vector<int> &nums) {
        ranges::sort(nums);
        int n = nums.size();
        int m = unique(nums.begin(), nums.end()) - nums.begin(); // 原地去重
        int ans = 0, left = 0;
        for (int i = 0; i < m; i++) {
            while (nums[left] < nums[i] - n + 1) { // nums[left] 不在窗口内
                left++;
            }
            ans = max(ans, i - left + 1);
        }
        return n - ans;
    }
};
//正难则反
//nums[left] < nums[i] - n + 1转换为nums[left] + n - 1 < nums[i]，left代表的就是左边界最小值，而nums[i]类比最大值，维护一个安全的窗口边界。
```

使用**unique**函数来去除nums容器（通常是一个std::vector或std::array等）中的重复元素。然后，它计算了去除重复元素后nums中剩余的唯一元素的数量，并将这个数量赋值给整数m。

具体来说：

1. `unique`函数接受两个迭代器参数，表示一个范围（在这里是`nums.begin()`到`nums.end()`），并重新排列该范围内的元素，使得所有重复的元素都出现在范围的末尾。它返回一个迭代器，指向最后一个唯一元素的下一个位置。
2. 通过将`std::unique`返回的迭代器减去`nums.begin()`，我们得到了唯一元素的数量。这是因为迭代器之间的差值是它们之间的元素数量。
3. 这个数量被赋值给整数`m`。

需要注意的是，`std::unique`函数并不真正地从容器中删除任何元素，它只是重新排列元素，使得所有唯一的元素都出现在前面，而重复的元素都出现在后面。如果你想真正地删除这些重复的元素，你通常需要结合使用`nums.erase()`方法，像这样：

```cpp
nums.erase(unique(nums.begin(), nums.end()), nums.end());
```

这样，`nums`中的重复元素就会被真正地从容器中删除了。

## 分组循环



## 双指针

### [2825. 循环增长使字符串子序列等于另一个字符串](https://leetcode.cn/problems/make-string-a-subsequence-using-cyclic-increments/)

```c++
class Solution {  
public:  
    // 判断字符串s是否是字符串t的子序列  
    bool canMakeSubsequence(string s, string t) {  //str1：s，str2：t
        // 如果s的长度小于t的长度，t不可能是s的子序列  
        if (s.length() < t.length()) {    
            return false;    
        }    
        // j用于追踪t中当前需要匹配的字符位置  
        int j = 0;    
        // 遍历s中的每个字符  
        for (char b : s) {    
            // 计算b的下一个字符，如果是'z'则回环到'a'  
            char c = (b != 'z') ? (b + 1) : 'a';//用另一个变量保存提前操作量    
            // 如果b或c等于t中当前位置的字符  
            if (b == t[j] || c == t[j]) {    
                // 向前移动t中当前位置指针  
                 j++;      
                // 如果t中的字符已经完全匹配完（即j等于t的长度）  
                if (j == t.length()) {    
                    // 返回true，表示t是s的子序列  
                    return true;    
                }    
            }  
        }   
        return false;    
    }    
};
```

题目提到至多一次完成操作，即可知道**一次遍历**后，若遍历子序列的指针长度不等于子序列说明字符不匹配。

##  贪心算法

### [3111. 覆盖所有点的最少矩形数目](https://leetcode.cn/problems/minimum-rectangles-to-cover-points/)

```c++
class Solution {
public:
    static bool cmp(const vector<int>&a, const vector<int>&b){
        return a[0] < b[0];
    }
    int minRectanglesToCoverPoints(vector<vector<int>>& points, int w) {
        sort(points.begin(), points.end(), cmp);
        int res = 0;
        int pre = -1;
        for(int i = 0; i < points.size(); i++){
            if(points[i][0] > pre){
                res++;
                pre = points[i][0] + w;//更新左边界
            }
        }
        return res;
    }
};
```

### [3106. 满足距离约束且字典序最小的字符串](https://leetcode.cn/problems/lexicographically-smallest-string-after-operations-with-constraint/)

```c++
class Solution {  
public:  
    string getSmallestString(string s, int k) {  
        for (int i = 0; i < s.length(); i++) {  
            // 计算当前字符s[i]到'a'的距离和到'z'的距离，取两者中的较小值  
            // 这是因为字符集是循环的，我们需要考虑'z'之后是'a'的情况  
            int dis = min(s[i] - 'a', 'z' - s[i] + 1);         
            // 如果当前字符到'a'或'z'的最小距离大于k，即无法将s[i]变为'a'  
            if (dis > k) {  
                // 尽可能地减小s[i]，即s[i] -= k  
                s[i] -= k;  
                // 跳出循环，因为剩余的操作次数已经不足以改变后面的字符了  
                break;  
            }                
            // 如果可以将s[i]变为'a'，则将其变为'a' ,以达到最优 
            s[i] = 'a';  
            // 更新剩余操作次数，减去当前字符到'a'的距离  
            k -= dis;  
        }            
        // 返回处理后的字符串s  
        return s;  
    }  
};
```

# 模板算法

## 排序

**稳定性：**同样大小的样本在排序之后不会改变原始的相对次序。

**重要排序算法的总结：**

数据量非常小的情况下可以做到非常迅速：**插入排序**

性能优异、实现简单且利于改进（面对不同业务可以选择不同划分策略）、不在乎稳定性：**随机快排**

性能优异、不在乎额外空间占用、具有稳定性：**归并排序**

性能优异、额外空间占用要求O(1)、不在乎稳定性：**堆排序**

![image-20240924235425680](C:/Users/ka'wa/AppData/Roaming/Typora/typora-user-images/image-20240924235425680.png)

### 归并排序

**特点**：

- **分治法**：将数组分为两个部分，递归地排序这两个部分，然后合并已排序的部分。
- **合并过程**：通过临时数组将两个已排序数组合并，确保最终结果也是有序的。
- **时间复杂度**：最优、最坏和平均情况均为 `O(n log n)`。
- **空间复杂度**：`O(n)`，需要额外的存储空间。
- **稳定性**：是稳定的排序算法，保持相等元素的相对顺序。

```c++
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 1e5 + 1;

ll help[N], a[N];
void merge(ll l, ll r){
    //1、将数组排序
    if(l >= r) return;

    ll mid = l + ((r - l) >> 1);
    ll i = l, j = mid + 1, t = l;
    while(i <= mid && j <= r){
        help[t++] = a[i] <= a[j] ? a[i++] : a[j++];
    }
    //2、将左半区域剩余元素排序
    while(i <= mid){
        help[t++] = a[i++];
    }
    //3、将右半区域剩余元素排序
    while(j <= r){
        help[t++] = a[j++];
    }
    //4、复制数组
    for(int i = l; i <= r; i++){
        a[i] = help[i];
    }
    return;
}

void mergeSort(ll l, ll r){
    if(l < r){
        ll mid = l + ((r - l) >> 1);
        //左边区域
        mergeSort(l, mid);
        //右边区域
        mergeSort(mid + 1, r);
        //合并左右区域
        merge(l, r);
    }
}

int main() {
    ios::sync_with_stdio(0); cin.tie(0);
    cin.tie(0),cout.tie(0);

    ll n;
    cin >> n;

    for(int i = 0; i < n; i++){
        cin >> a[i];
    }

    mergeSort(0, n - 1);
    for(int i = 0; i < n; i++){
        cout << a[i] << " ";
    }

    return 0;
}
```

### 随机快速排序

**特点**：

- **分治法**：通过选择一个“基准”元素，将数组分成两个部分，递归地排序这两个部分。
- **随机性**：随机选择基准元素，可以减少最坏情况出现的概率（例如，在已经排好序的数组中，快速排序会退化为 `O(n^2)`，而随机选择可以避免这个问题）。
- **时间复杂度**：最优和平均情况为 `O(n log n)`，最坏情况为 `O(n^2)`。
- **空间复杂度**：`O(log n)`，递归栈空间。
- **稳定性**：是不稳定的排序算法，可能改变相等元素的相对顺序。

随机生成数：

```c++
int randomIndex = rand() % (r - l + 1) + l;//随机选择r 到 l之间的一个数
srand(time(nullptr)); // 初始化随机数生成器,在main函数中定义
```



```c++
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 1e5 + 1;

ll arr[5] = {9,7,5,4,2};
int first, last;
//交换数值
void swap(int i, int j) {
	ll tmp = arr[i];
	arr[i] = arr[j];
	arr[j] = tmp;
}

//荷兰国旗问题(优化随机快排)
//优化点：选出一个x，在划分的时候搞定所有值是x的数字
void pattion2(int l, int r, int x) {
	first = l;
	last = r;
	int i = l;
	while (i <= last) {
		if (arr[i] < x) {
			swap(first, i);
			first++;
			i++;
		}
		else if (arr[i] == x) {
			i++;
		}
		else{
			swap(last, i);
			last--;
		}
	}
}


void QuickSort2(int l, int r) {
	if (l < r) {
		int QIndex = rand() % (r - l + 1) + l;//随机下标
		int pvit = arr[QIndex];//基准值
		pattion2(l, r, pvit);
        //采用临时变量记录
		int left = first;
		int right = last;
		QuickSort1(l, left - 1);
		QuickSort1(right + 1, r);
	}
}

int main() {
	int n = 5;
	//cin >> n;
	//
	//for (int i = 0; i < n; i++) {
	//	cin >> arr[i];
	//}

	srand(time(nullptr));
	QuickSort2(0, n - 1);

	for (int i = 0; i < n; i++) {
		cout << arr[i] << " ";
	}

	return 0;
}
```

### 堆排序

**特点**：

- **堆数据结构**：利用堆（通常是最大堆）结构来进行排序。最大堆保证了父节点大于或等于其子节点。
- **分两步**：首先构建一个最大堆，然后不断将堆顶元素（最大值）与未排序部分的最后一个元素交换，并重新调整堆。
- **时间复杂度**：最优、最坏和平均情况均为 `O(n log n)`。
- **空间复杂度**：`O(1)`，原地排序。
- **稳定性**：是不稳定的排序算法，可能改变相等元素的相对顺序。

```c++
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 1e5 + 1;

//堆结构(将子树中最大的值放到顶点)
//父亲节点：（i - 1） / 2
//左孩子节点：i * 2 + 1
//右孩子节点：i * 2 + 2 

void swap(vector<int>arr, int i, int j) {
	int tmp = arr[i];
	arr[i] = arr[j];
	arr[j] = tmp;
}


//比较头节点,向上调整大根堆
void heapinsert(vector<int>arr, int i) {
	while (arr[i] > arr[(i - 1) / 2]) {
		swap(arr, i, (i - 1) / 2);
		i = (i - 1) / 2;
	}
}

//比较左右孩子节点，向下调整大根堆
void heapify(vector<int>arr, int i, int size) {
	int l = i * 2 + 1;
	while (l < size) {
		//有右孩子，l + 1
		//左孩子，l
		//最强的孩子是哪个下标
		int best = l + 1 < size && arr[l + 1] > arr[l] ? l + 1 : l;
		//最强的下标是谁
		best = arr[best] > arr[i] ? best : i;
		if (best == i) {
			break;
		}
		swap(arr, best, i);
		i = best;
		l = i * 2 + 1;
	}
}


//从顶到底建立大根堆O(N * logN)
//依次弹出堆内最大值并排序O(N * logN)
void heapsort1(vector<int>arr) {
	int n = arr.size();
	for (int i = 0; i < n; i++) {
		heapinsert(arr, i);
	}
	int size = n;
	while (size > 1) {
		swap(arr, 0, --size);
		heapify(arr, 0, size);
	}
}


//从底到顶建立大根堆O（N）
//依次弹出堆内最大值排序O(N * logN)
void heapsort2(vector<int>arr) {
	int n = arr.size();
	for (int i = 0; i < n; i++) {
		heapify(arr, i, n);
	}
	int size = n;
	while (size > 1) {
		swap(arr, 0, --size);
		heapify(arr, 0, size);
	}
}

int main() {

}
```

### 基数排序

概念：排序数组中非负整数

* **特点**：
	- **按位排序**：将数字分解为多个数字，根据每个位（如个位、十位、百位等）进行排序。
	- **利用计数排序**：通常结合计数排序进行每位的排序，以确保稳定性。
	- **时间复杂度**：最佳、最坏和平均情况均为 `O(d * (n + k))`，其中 `d` 是数字的位数，`k` 是基数（桶的数量）。
	- **空间复杂度**：`O(n + k)`，需要额外的存储空间。
	- **稳定性**：是稳定的排序算法，保持相等元素的相对顺序。

```c++
#include<bits/stdc++.h>
using namespace std;

typedef long long ll;
//假如BASE = 1000，那么999就是一位
const int BASE = 10;//10进制


void radix_sort(vector<int>& arr) {
	int max = *max_element(arr.begin(), arr.end());
	// 获取最大数的位数d
	int d = 0;
	while (max) {
		max /= 10;
		d++;
	}

	int* count = new int[10];  // 计数器，也就是0~9共10个桶 
	int* tem = new int[arr.size()];  // 临时数组，和计数排序的临时数组作用一样 

	int radix = 1;
	for (int i = 0; i < d; radix *= BASE, i++) {// 可以看成进行了d次计数排序，以下代码和计数排序万分相像 
		// 每次将计数器清零
		for (int j = 0; j < 10; j++) {
			count[j] = 0;
		}
		for (int j = 0; j < arr.size(); j++) {
			// 计数，方便后续获得每个数的index
			// 频次统计 
			count[(arr[j] / radix) % 10]++;
		}
		//前缀统计频次
		for (int j = 1; j < 10; j++) {
			count[j] += count[j - 1];
		}
		//逆序进行排序
		for (int j = arr.size() - 1; j >= 0; j--) {
			// 将桶里的元素取出来 
			int index = count[(arr[j] / radix) % 10] - 1;
			tem[index] = arr[j];
			count[(arr[j] / radix) % 10]--;
		}
		for (int j = 0; j < arr.size(); j++) {
			arr[j] = tem[j];
		}
	}

}
int main() {

	vector<int> arr = { 61, 17, 29, 22, 34, 60, 72, 21, 50, 1, 62 };

	radix_sort(arr);

	for (int x : arr) {
		cout << x << " ";
	}

	return 0;
}
```

### 简单排序

#### 插入排序：

**特点**：

- **插入过程**：将未排序的元素逐一插入到已排序部分的合适位置。
- **逐步构建**：在每一步中，假设前面部分已经排序，将当前元素插入到其适当位置。
- **时间复杂度**：最优情况为 `O(n)`（对于已排序数组），最坏和平均情况均为 `O(n^2)`。
- **空间复杂度**：`O(1)`，原地排序。
- **稳定性**：是稳定的排序算法，保持相等元素的相对顺序。

```c++
#include<bits/stdc++.h>
using namespace std;

// 插入排序函数
void insertionSort(std::vector<int>& arr) {
    int n = arr.size();
    // 从第1个元素开始，逐个向右插入
    for (int i = 1; i < n; i++) {
        int key = arr[i];  // 记录当前要插入的元素
        int j = i - 1;
        // 向左遍历已排序的部分，找到合适的位置插入
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];  // 将大于key的元素向右移动
            j--;
        }
        arr[j + 1] = key;  // 找到合适的位置，将key插入
    }
}

int main() {
    // 定义一个无序数组
    vector<int> arr = {12, 11, 13, 5, 6};

    cout << "未排序的数组: ";
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;

    // 调用插入排序
    insertionSort(arr);

    cout << "插入排序后的数组: ";
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;

    return 0;
}

```

#### 选择排序：

**特点**：

- **选择过程**：每一次迭代从未排序部分中选择最小（或最大）元素，将其放到已排序部分的末尾。
- **时间复杂度**：最优、最坏和平均情况均为 `O(n^2)`。
- **空间复杂度**：`O(1)`，原地排序。
- **稳定性**：是不稳定的排序算法，可能改变相等元素的相对顺序。

```c++
#include<bits/stdc++.h>
using namespace std;

// 选择排序函数
void selectionSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i;  // 假设当前元素为最小值
        // 在未排序部分中找到最小值
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIndex]) {
                minIndex = j;  // 记录最小值的索引
            }
        }
        // 将最小值与当前元素交换
        swap(arr[i], arr[minIndex]);
    }
}

int main() {
    // 定义一个无序数组
    vector<int> arr = {64, 25, 12, 22, 11};

    cout << "未排序的数组: ";
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;

    // 调用选择排序
    selectionSort(arr);

    cout << "选择排序后的数组: ";
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;

    return 0;
}
```

### 冒泡排序

**特点**：

- **逐步比较**：通过重复遍历待排序的元素，比较相邻的两个元素，如果顺序错误则交换它们。
- **冒泡效果**：较大的元素通过交换“冒泡”到数组的末尾。
- **时间复杂度**：最优和最坏情况均为 `O(n^2)`，平均情况也为 `O(n^2)`。
- **空间复杂度**：`O(1)`，原地排序。
- **稳定性**：是稳定的排序算法，保持相等元素的相对顺序。

```c++
#include <iostream>
#include <vector>

using namespace std;

// 冒泡排序函数
void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);  // 交换相邻元素
            }
       }
    }
}

int main() {
    // 定义一个无序数组
    vector<int> arr = {64, 34, 25, 12, 22, 11, 90};

    cout << "未排序的数组: ";
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;

    // 调用冒泡排序
    bubbleSort(arr);

    cout << "冒泡排序后的数组: ";
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;

    return 0;
}
```

### 希尔排序

**特点**：

- **分组插入**：将数组分成多个子数组，使用增量对每个子数组进行插入排序。
- **逐步减少增量**：随着增量逐渐减少，最后增量为1时进行最终的插入排序。
- **时间复杂度**：最优情况为 `O(n log n)`，最坏情况为 `O(n^2)`（具体依赖于增量序列）。
- **空间复杂度**：`O(1)`，原地排序。
- **稳定性**：是不稳定的排序算法，可能改变相等元素的相对顺序。

```c++
#include <iostream>
#include <vector>

using namespace std;

// 希尔排序函数
void shellSort(vector<int>& arr) {
    int n = arr.size();
    // 初始化增量
    for (int gap = n / 2; gap > 0; gap /= 2) {
        // 对每个增量进行插入排序
        for (int i = gap; i < n; i++) {
            int temp = arr[i];  // 暂存当前元素
            int j;
            // 将当前元素插入到合适位置
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
                arr[j] = arr[j - gap];  // 元素向右移动
            }
            arr[j] = temp;  // 插入元素
        }
    }
}

int main() {
    // 定义一个无序数组
    vector<int> arr = {12, 34, 54, 2, 3};

    cout << "未排序的数组: ";
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;

    // 调用希尔排序
    shellSort(arr);

    cout << "希尔排序后的数组: ";
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;

    return 0;
}
```



## 前缀树

前缀树是一种功能强大且灵活的字符串数据结构，适合处理大量字符串的查找、插入和前缀相关操作。

```c++
#include <bits/stdc++.h> 
using namespace std;

const int MAXN = 150001;  // 最大节点数。Trie 树总共有 MAXN 个节点，每个节点最多有26个孩子

int tree[MAXN][26];    // Trie 树的节点，二维数组表示每个节点的 26 个英文字母分支（0-25表示a-z）
int endCount[MAXN];    // 记录某个节点是否是一个单词的结尾。endCount[i] 表示以该节点为结尾的单词数量
int pass[MAXN];        // 记录经过某个节点的次数，pass[i] 表示通过该节点的单词数
int cnt;               // 当前 Trie 树的节点计数，初始时只有根节点，编号为 1

// 初始化 Trie 树，将根节点编号设置为 1
void build() {
    cnt = 1;  // 根节点编号为 1，cnt 表示当前的节点数量
    memset(tree, 0, sizeof(tree));      // 清空 tree 数组，表示 Trie 树中没有任何连接
    memset(endCount, 0, sizeof(endCount));  // 清空 endCount 数组
    memset(pass, 0, sizeof(pass));      // 清空 pass 数组
}

// 将一个单词插入到 Trie 树中
void insert(const string& word) {
    int cur = 1;  // 当前节点，1 表示从根节点开始
    pass[cur]++;  // 根节点被访问一次
    for (int i = 0, path; i < word.length(); i++) {
        path = word[i] - 'a';  // 计算字符的位置，'a' 对应 0，'b' 对应 1，以此类推
        if (tree[cur][path] == 0) {  // 如果当前节点的该路径不存在，创建新节点
            tree[cur][path] = ++cnt;  // 分配一个新节点编号，cnt 自增
        }
        cur = tree[cur][path];  // 移动到下一个节点
        pass[cur]++;  // 更新新节点的 pass 值，表示通过该节点
    }
    endCount[cur]++;  // 该节点是单词的结尾，更新 endCount，表示这里有一个单词结束
}

// 查找某个单词是否存在于 Trie 树中
int search(const string& word) {
    int cur = 1;  // 从根节点开始搜索
    for (int i = 0, path; i < word.length(); i++) {
        path = word[i] - 'a';  // 计算当前字符对应的路径
        if (tree[cur][path] == 0) {  // 如果路径不存在，说明该单词不在 Trie 树中
            return 0;
        }
        cur = tree[cur][path];  // 继续向下搜索
    }
    return endCount[cur];  // 返回是否有单词以该节点结尾，>0 表示存在该单词
}

// 返回以某个前缀开头的单词数量
int prefixNumber(const string& pre) {
    int cur = 1;  // 从根节点开始搜索前缀
    for (int i = 0, path; i < pre.length(); i++) {
        path = pre[i] - 'a';  // 计算当前字符对应的路径
        if (tree[cur][path] == 0) {  // 如果路径不存在，说明没有单词以该前缀开头
            return 0;
        }
        cur = tree[cur][path];  // 继续向下搜索
    }
    return pass[cur];  // 返回经过该前缀节点的单词数量
}

// 删除 Trie 树中的某个单词
void deleteWord(const string& word) {
    if (search(word) > 0) {  // 先判断该单词是否存在，如果不存在，不需要删除
        int cur = 1;  // 从根节点开始删除
        for (int i = 0, path; i < word.length(); i++) {
            path = word[i] - 'a';  // 计算当前字符的路径
            if (--pass[tree[cur][path]] == 0) {  // 如果删除后路径上没有单词经过，直接删除该路径
                tree[cur][path] = 0;  // 将路径置为 0，表示该路径不再存在
                return;  // 直接退出
            }
            cur = tree[cur][path];  // 继续向下处理
        }
        endCount[cur]--;  // 减少该节点的 endCount 值，表示该单词被删除
    }
}

// 清空整个 Trie 树
void clear() {
    for (int i = 1; i <= cnt; i++) {  // 遍历所有节点
        memset(tree[i], 0, sizeof(tree[i]));  // 清空每个节点的路径
        endCount[i] = 0;  // 清空单词结束标记
        pass[i] = 0;      // 清空经过节点的单词数量
    }
}

// 主函数
int main() {
    ios::sync_with_stdio(false);  // 提高输入输出效率
    cin.tie(nullptr);  // 解除输入输出流的绑定

    int m, op;  // m 是操作数，op 是操作类型
    string line;  // 用于存储输入行
    while (getline(cin, line)) {  // 读取输入行
        build();  // 初始化 Trie 树
        m = stoi(line);  // 将字符串转换为整数，表示操作数
        for (int i = 1; i <= m; i++) {
            getline(cin, line);  // 读取操作行
            size_t space_pos = line.find(' ');  // 找到操作与单词之间的空格位置
            op = stoi(line.substr(0, space_pos));  // 解析操作类型
            string word = line.substr(space_pos + 1);  // 解析单词

            if (op == 1) {
                insert(word);  // 插入单词
            } else if (op == 2) {
                deleteWord(word);  // 删除单词
            } else if (op == 3) {
                cout << (search(word) > 0 ? "YES" : "NO") << endl;  // 判断单词是否存在
            } else if (op == 4) {
                cout << prefixNumber(word) << endl;  // 返回以该前缀开头的单词数量
            }
        }
        clear();  // 清空 Trie 树
    }
    
    return 0;  // 正常退出程序
}
```

# STL- 常用容器

## string容器

### string基本概念

**本质：**

- string是C++风格的字符串，而string本质上是一个类

**string和char \* 区别：**

- char * 是一个指针
- string是一个类，类内部封装了char* ，管理这个字符串，是一个char*型的容器

**特点：**

string 类内部封装了很多成员方法

例如：查找find，拷贝copy，删除delete 替换replace，插入insert

string管理char*所分配的内存，不用担心复制越界和取值越界等，由类内部进行负责

### string构造函数

构造函数原型：

- `string();` //创建一个空的字符串 例如: string str;
	`string(const char* s);` //使用字符串s初始化
- `string(const string& str);` //使用一个string对象初始化另一个string对象
- `string(int n, char c);` //使用n个字符c初始化

**示例：**

```c++
#include <string>
//string构造
void test01()
{
	string s1; //创建空字符串，调用无参构造函数
	cout << "str1 = " << s1 << endl;

	const char* str = "hello world";
	string s2(str); //把c_string转换成了string

	cout << "str2 = " << s2 << endl;

	string s3(s2); //调用拷贝构造函数
	cout << "str3 = " << s3 << endl;

	string s4(10, 'a');
	cout << "str3 = " << s3 << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：string的多种构造方式没有可比性，灵活使用即可

### string赋值操作

功能描述：

- 给string字符串进行赋值

赋值的函数原型：

- `string& operator=(const char* s);` //char*类型字符串 赋值给当前的字符串
- `string& operator=(const string &s);` //把字符串s赋给当前的字符串
- `string& operator=(char c);` //字符赋值给当前的字符串
- `string& assign(const char *s);` //把字符串s赋给当前的字符串
- `string& assign(const char *s, int n);` //把字符串s的前n个字符赋给当前的字符串
- `string& assign(const string &s);` //把字符串s赋给当前字符串
- `string& assign(int n, char c);` //用n个字符c赋给当前字符串

**示例：**

```c++
C++

//赋值
void test01()
{
	string str1;
	str1 = "hello world";
	cout << "str1 = " << str1 << endl;

	string str2;
	str2 = str1;
	cout << "str2 = " << str2 << endl;

	string str3;
	str3 = 'a';
	cout << "str3 = " << str3 << endl;

	string str4;
	str4.assign("hello c++");
	cout << "str4 = " << str4 << endl;

	string str5;
	str5.assign("hello c++",5);
	cout << "str5 = " << str5 << endl;


	string str6;
	str6.assign(str5);
	cout << "str6 = " << str6 << endl;

	string str7;
	str7.assign(5, 'x');
	cout << "str7 = " << str7 << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

 string的赋值方式很多，`operator=` 这种方式是比较实用的

### string字符串拼接

**功能描述：**

- 实现在字符串末尾拼接字符串

**函数原型：**

- `string& operator+=(const char* str);` //重载+=操作符
- `string& operator+=(const char c);` //重载+=操作符
- `string& operator+=(const string& str);` //重载+=操作符
- `string& append(const char *s); `//把字符串s连接到当前字符串结尾
- `string& append(const char *s, int n);` //把字符串s的前n个字符连接到当前字符串结尾
- `string& append(const string &s);` //同operator+=(const string& str)
- `string& append(const string &s, int pos, int n);`//字符串s中从pos开始的n个字符连接到字符串结尾

**示例：**

```c++
C++

//字符串拼接
void test01()
{
	string str1 = "我";

	str1 += "爱玩游戏";

	cout << "str1 = " << str1 << endl;
	
	str1 += ':';

	cout << "str1 = " << str1 << endl;

	string str2 = "LOL DNF";

	str1 += str2;

	cout << "str1 = " << str1 << endl;

	string str3 = "I";
	str3.append(" love ");
	str3.append("game abcde", 4);
	//str3.append(str2);
	str3.append(str2, 4, 3); // 从下标4位置开始 ，截取3个字符，拼接到字符串末尾
	cout << "str3 = " << str3 << endl;
}
int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：字符串拼接的重载版本很多，初学阶段记住几种即可

### string查找和替换

**功能描述：**

- 查找：查找指定字符串是否存在
- 替换：在指定的位置替换字符串

**函数原型：**

- `int find(const string& str, int pos = 0) const;` //查找str第一次出现位置,从pos开始查找
- `int find(const char* s, int pos = 0) const; `//查找s第一次出现位置,从pos开始查找
- `int find(const char* s, int pos, int n) const; `//从pos位置查找s的前n个字符第一次位置
- `int find(const char c, int pos = 0) const; `//查找字符c第一次出现位置
- `int rfind(const string& str, int pos = npos) const;` //查找str最后一次位置,从pos开始查找
- `int rfind(const char* s, int pos = npos) const;` //查找s最后一次出现位置,从pos开始查找
- `int rfind(const char* s, int pos, int n) const;` //从pos查找s的前n个字符最后一次位置
- `int rfind(const char c, int pos = 0) const; `//查找字符c最后一次出现位置
- `string& replace(int pos, int n, const string& str); `//替换从pos开始n个字符为字符串str
- `string& replace(int pos, int n,const char* s); `//替换从pos开始的n个字符为字符串s

**示例：**

```c++
C++

//查找和替换
void test01()
{
	//查找
	string str1 = "abcdefgde";

	int pos = str1.find("de");

	if (pos == -1)
	{
		cout << "未找到" << endl;
	}
	else
	{
		cout << "pos = " << pos << endl;
	}
	

	pos = str1.rfind("de");

	cout << "pos = " << pos << endl;

}

void test02()
{
	//替换
	string str1 = "abcdefgde";
    //从1位置替换“bcd”为“1111”
	str1.replace(1, 3, "1111");

	cout << "str1 = " << str1 << endl;
}

int main() {

	//test01();
	//test02();

	system("pause");

	return 0;
}
```

总结：

- find查找是从左往后，rfind从右往左
- find找到字符串后返回查找的第一个字符位置，找不到返回-1
- replace在替换时，要指定从哪个位置起，多少个字符，替换成什么样的字符串

### string字符串比较

**功能描述：**

- 字符串之间的比较

**比较方式：**

- 字符串比较是按字符的ASCII码进行对比

= 返回 0

\> 返回 1

< 返回 -1

**函数原型：**

- `int compare(const string &s) const; `//与字符串s比较
- `int compare(const char *s) const;` //与字符串s比较

**示例：**

```c++
C++

//字符串比较
void test01()
{

	string s1 = "hello";
	string s2 = "aello";

	int ret = s1.compare(s2);

	if (ret == 0) {
		cout << "s1 等于 s2" << endl;
	}
	else if (ret > 0)
	{
		cout << "s1 大于 s2" << endl;
	}
	else
	{
		cout << "s1 小于 s2" << endl;
	}

}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：字符串对比主要是用于比较两个字符串是否相等，判断谁大谁小的意义并不是很大

### string字符存取

string中单个字符存取方式有两种

- `char& operator[](int n); `//通过[]方式取字符
- `char& at(int n); `//通过at方法获取字符

**示例：**

```c++
C++

void test01()
{
	string str = "hello world";

	for (int i = 0; i < str.size(); i++)
	{
		cout << str[i] << " ";
	}
	cout << endl;

	for (int i = 0; i < str.size(); i++)
	{
		cout << str.at(i) << " ";
	}
	cout << endl;


	//字符修改
	str[0] = 'x';
	str.at(1) = 'x';
	cout << str << endl;
	
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：string字符串中单个字符存取有两种方式，利用 [ ] 或 at

### string插入和删除

**功能描述：**

- 对string字符串进行插入和删除字符操作

**函数原型：**

- `string& insert(int pos, const char* s); `//插入字符串
- `string& insert(int pos, const string& str); `//插入字符串
- `string& insert(int pos, int n, char c);` //在指定位置插入n个字符c
- `string& erase(int pos, int n = npos);` //删除从Pos开始的n个字符

**示例：**

```c++
C++

//字符串插入和删除
void test01()

	str.erase(1, 3);  //从1号位置开始3个字符
	cout << str << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

**总结：**插入和删除的起始下标都是从0开始

### string子串

**功能描述：**

- 从字符串中获取想要的子串

**函数原型：**

- `string substr(int pos = 0, int n = npos) const;` //返回由pos开始的n个字符组成的字符串

**示例：**

```c++
C++

//子串
void test01()
{

	string str = "abcdefg";
	string subStr = str.substr(1, 3);
	cout << "subStr = " << subStr << endl;

	string email = "hello@sina.com";
	int pos = email.find("@");
	string username = email.substr(0, pos);
	cout << "username: " << username << endl;

}

int main() {

	test01();

	system("pause");

	return 0;
}
```

**总结：**灵活的运用求子串功能，可以在实际开发中获取有效的信息

## vector容器

### vector基本概念

**功能：**

- vector数据结构和**数组非常相似**，也称为**单端数组**

**vector与普通数组区别：**

- 不同之处在于数组是静态空间，而vector可以**动态扩展**

**动态扩展：**

- 并不是在原空间之后续接新空间，而是找更大的内存空间，然后将原数据拷贝新空间，释放原空间

[![说明: 2015-11-10_151152](https://pic.imgdb.cn/item/61dd2d832ab3f51d9176c412.jpg)](https://pic.imgdb.cn/item/61dd2d832ab3f51d9176c412.jpg)

- vector容器的迭代器是支持随机访问的迭代器

### vector构造函数

**功能描述：**

- 创建vector容器

**函数原型：**

- `vector<T> v; `//采用模板实现类实现，默认构造函数
- `vector(v.begin(), v.end()); `//将v[begin(), end())区间中的元素拷贝给本身。
- `vector(n, elem);` //构造函数将n个elem拷贝给本身。
- `vector(const vector &vec);` //拷贝构造函数。

**示例：**

```c++
C++

#include <vector>

void printVector(vector<int>& v) {

	for (vector<int>::iterator it = v.begin(); it != v.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}

void test01()
{
	vector<int> v1; //无参构造
	for (int i = 0; i < 10; i++)
	{
		v1.push_back(i);
	}
	printVector(v1);

	vector<int> v2(v1.begin(), v1.end());
	printVector(v2);

	vector<int> v3(10, 100);
	printVector(v3);
	
	vector<int> v4(v3);
	printVector(v4);
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

**总结：**vector的多种构造方式没有可比性，灵活使用即可

### vector赋值操作

**功能描述：**

- 给vector容器进行赋值

**函数原型：**

- `vector& operator=(const vector &vec);`//重载等号操作符
- `assign(begin, end);` //将[beg, end)区间中的数据拷贝赋值给本身。
- `assign(n, elem);` //将n个elem拷贝赋值给本身。

**示例：**

```c++
C++

#include <vector>
void printVector(vector<int>& v) {

	for (vector<int>::iterator it = v.begin(); it != v.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}

//赋值操作
void test01()
{
	vector<int> v1; //无参构造
	for (int i = 0; i < 10; i++)
	{
		v1.push_back(i);
	}
	printVector(v1);

	vector<int>v2;
	v2 = v1;
	printVector(v2);

	vector<int>v3;
	v3.assign(v1.begin(), v1.end());
	printVector(v3);

	vector<int>v4;
	v4.assign(10, 100);
	printVector(v4);
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结： vector赋值方式比较简单，使用operator=，或者assign都可以

### vector容量和大小

**功能描述：**

- 对vector容器的容量和大小操作

**函数原型：**

- `empty(); `//判断容器是否为空

- `capacity();` //容器的容量

- `size();` //返回容器中元素的个数

- `resize(int num);` //重新指定容器的长度为num，若容器变长，则以默认值填充新位置。

	//如果容器变短，则末尾超出容器长度的元素被删除。

- `resize(int num, elem);` //重新指定容器的长度为num，若容器变长，则以elem值填充新位置。

	//如果容器变短，则末尾超出容器长度的元素被删除

**示例：**

```c++
C++

#include <vector>

void printVector(vector<int>& v) {

	for (vector<int>::iterator it = v.begin(); it != v.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}

void test01()
{
	vector<int> v1;
	for (int i = 0; i < 10; i++)
	{
		v1.push_back(i);
	}
	printVector(v1);
	if (v1.empty())
	{
		cout << "v1为空" << endl;
	}
	else
	{
		cout << "v1不为空" << endl;
		cout << "v1的容量 = " << v1.capacity() << endl;
		cout << "v1的大小 = " << v1.size() << endl;
	}

	//resize 重新指定大小 ，若指定的更大，默认用0填充新位置，可以利用重载版本替换默认填充
	v1.resize(15,10);
	printVector(v1);

	//resize 重新指定大小 ，若指定的更小，超出部分元素被删除
	v1.resize(5);
	printVector(v1);
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

- 判断是否为空 — empty
- 返回元素个数 — size
- 返回容器容量 — capacity
- 重新指定大小 — resize

### vector插入和删除

**功能描述：**

- 对vector容器进行插入、删除操作

**函数原型：**

- `push_back(ele);` //尾部插入元素ele
- `pop_back();` //删除最后一个元素
- `insert(const_iterator pos, ele);` //迭代器指向位置pos插入元素ele
- `insert(const_iterator pos, int count,ele);`//迭代器指向位置pos插入count个元素ele
- `erase(const_iterator pos);` //删除迭代器指向的元素
- `erase(const_iterator start, const_iterator end);`//删除迭代器从start到end之间的元素
- `clear();` //删除容器中所有元素

**示例：**

```c++
C++

#include <vector>

void printVector(vector<int>& v) {

	for (vector<int>::iterator it = v.begin(); it != v.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}

//插入和删除
void test01()
{
	vector<int> v1;
	//尾插
	v1.push_back(10);
	v1.push_back(20);
	v1.push_back(30);
	v1.push_back(40);
	v1.push_back(50);
	printVector(v1);
	//尾删
	v1.pop_back();
	printVector(v1);
	//插入
	v1.insert(v1.begin(), 100);
	printVector(v1);

	v1.insert(v1.begin(), 2, 1000);
	printVector(v1);

	//删除
	v1.erase(v1.begin());
	printVector(v1);

	//清空
	v1.erase(v1.begin(), v1.end());
	v1.clear();
	printVector(v1);
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

- 尾插 — push_back
- 尾删 — pop_back
- 插入 — insert (位置迭代器)
- 删除 — erase （位置迭代器）
- 清空 — clear

### vector数据存取

**功能描述：**

- 对vector中的数据的存取操作

**函数原型：**

- `at(int idx); `//返回索引idx所指的数据
- `operator[]; `//返回索引idx所指的数据
- `front(); `//返回容器中第一个数据元素
- `back();` //返回容器中最后一个数据元素

**示例：**

```c++
C++

#include <vector>

void test01()
{
	vector<int>v1;
	for (int i = 0; i < 10; i++)
	{
		v1.push_back(i);
	}

	for (int i = 0; i < v1.size(); i++)
	{
		cout << v1[i] << " ";
	}
	cout << endl;

	for (int i = 0; i < v1.size(); i++)
	{
		cout << v1.at(i) << " ";
	}
	cout << endl;

	cout << "v1的第一个元素为： " << v1.front() << endl;
	cout << "v1的最后一个元素为： " << v1.back() << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

- 除了用迭代器获取vector容器中元素，[ ]和at也可以
- front返回容器第一个元素
- back返回容器最后一个元素

### vector互换容器

**功能描述：**

- 实现两个容器内元素进行互换

**函数原型：**

- `swap(vec);` // 将vec与本身的元素互换

**示例：**

```c++
C++

#include <vector>
void printVector(vector<int>& v) {

	for (vector<int>::iterator it = v.begin(); it != v.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}

void test01()
{
	vector<int>v1;
	for (int i = 0; i < 10; i++)
	{
		v1.push_back(i);
	}
	printVector(v1);

	vector<int>v2;
	for (int i = 10; i > 0; i--)
	{
		v2.push_back(i);
	}
	printVector(v2);

	//互换容器
	cout << "互换后" << endl;
	v1.swap(v2);
	printVector(v1);
	printVector(v2);
}

void test02()
{
	vector<int> v;
	for (int i = 0; i < 100000; i++) {
		v.push_back(i);
	}

	cout << "v的容量为：" << v.capacity() << endl;
	cout << "v的大小为：" << v.size() << endl;

	v.resize(3);

	cout << "v的容量为：" << v.capacity() << endl;
	cout << "v的大小为：" << v.size() << endl;

	//收缩内存
	vector<int>(v).swap(v); //匿名对象

	cout << "v的容量为：" << v.capacity() << endl;
	cout << "v的大小为：" << v.size() << endl;
}

int main() {

	test01();

	test02();

	system("pause");

	return 0;
}
```

总结：swap可以使两个容器互换，可以达到实用的收缩内存效果

### vector预留空间

**功能描述：**

- 减少vector在动态扩展容量时的扩展次数

**函数原型：**

- `reserve(int len);`//容器预留len个元素长度，预留位置不初始化，元素不可访问。

**示例：**

```c++
C++

#include <vector>
void test01()
{
	vector<int> v;

	//预留空间
	v.reserve(100000);

	int num = 0;
	int* p = NULL;
	for (int i = 0; i < 100000; i++) {
		v.push_back(i);
		if (p != &v[0]) {
			p = &v[0];
			num++;
		}
	}

	cout << "num:" << num << endl;
}

int main() {

	test01();
    
	system("pause");

	return 0;
}
```

总结：如果数据量较大，可以一开始利用reserve预留空间

## deque容器

### deque容器基本概念

**功能：**

- 双端数组，可以对头端进行插入删除操作

**deque与vector区别：**

- vector对于头部的插入删除效率低，数据量越大，效率越低
- deque相对而言，对头部的插入删除速度回比vector快
- vector访问元素时的速度会比deque快,这和两者内部实现有关

[![说明: 2015-11-19_204101](https://pic.imgdb.cn/item/61dd2e922ab3f51d91779a78.jpg)](https://pic.imgdb.cn/item/61dd2e922ab3f51d91779a78.jpg)

deque内部工作原理:

deque内部有个**中控器**，维护每段缓冲区中的内容，缓冲区中存放真实数据

中控器维护的是每个缓冲区的地址，使得使用deque时像一片连续的内存空间

[![clip_image002-1547547896341](https://pic.imgdb.cn/item/61dd2eb22ab3f51d9177b488.jpg)](https://pic.imgdb.cn/item/61dd2eb22ab3f51d9177b488.jpg)

- deque容器的迭代器也是支持随机访问的

### deque构造函数

**功能描述：**

- deque容器构造

**函数原型：**

- `deque<T>` deqT; //默认构造形式
- `deque(beg, end);` //构造函数将[beg, end)区间中的元素拷贝给本身。
- `deque(n, elem);` //构造函数将n个elem拷贝给本身。
- `deque(const deque &deq);` //拷贝构造函数

**示例：**

```c++
C++

#include <deque>

void printDeque(const deque<int>& d) 
{
	for (deque<int>::const_iterator it = d.begin(); it != d.end(); it++) {
		cout << *it << " ";

	}
	cout << endl;
}
//deque构造
void test01() {

	deque<int> d1; //无参构造函数
	for (int i = 0; i < 10; i++)
	{
		d1.push_back(i);
	}
	printDeque(d1);
	deque<int> d2(d1.begin(),d1.end());
	printDeque(d2);

	deque<int>d3(10,100);
	printDeque(d3);

	deque<int>d4 = d3;
	printDeque(d4);
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

**总结：**deque容器和vector容器的构造方式几乎一致，灵活使用即可

### deque赋值操作

**功能描述：**

- 给deque容器进行赋值

**函数原型：**

- `deque& operator=(const deque &deq); `//重载等号操作符
- `assign(beg, end);` //将[beg, end)区间中的数据拷贝赋值给本身。
- `assign(n, elem);` //将n个elem拷贝赋值给本身。

**示例：**

```c++
C++

#include <deque>

void printDeque(const deque<int>& d) 
{
	for (deque<int>::const_iterator it = d.begin(); it != d.end(); it++) {
		cout << *it << " ";

	}
	cout << endl;
}
//赋值操作
void test01()
{
	deque<int> d1;
	for (int i = 0; i < 10; i++)
	{
		d1.push_back(i);
	}
	printDeque(d1);

	deque<int>d2;
	d2 = d1;
	printDeque(d2);

	deque<int>d3;
	d3.assign(d1.begin(), d1.end());
	printDeque(d3);

	deque<int>d4;
	d4.assign(10, 100);
	printDeque(d4);

}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：deque赋值操作也与vector相同，需熟练掌握

### deque大小操作

**功能描述：**

- 对deque容器的大小进行操作

**函数原型：**

- `deque.empty();` //判断容器是否为空

- `deque.size();` //返回容器中元素的个数

- `deque.resize(num);` //重新指定容器的长度为num,若容器变长，则以默认值填充新位置。

	//如果容器变短，则末尾超出容器长度的元素被删除。

- `deque.resize(num, elem);` //重新指定容器的长度为num,若容器变长，则以elem值填充新位置。

	//如果容器变短，则末尾超出容器长度的元素被删除。

**示例：**

```c++
C++

#include <deque>

void printDeque(const deque<int>& d) 
{
	for (deque<int>::const_iterator it = d.begin(); it != d.end(); it++) {
		cout << *it << " ";

	}
	cout << endl;
}

//大小操作
void test01()
{
	deque<int> d1;
	for (int i = 0; i < 10; i++)
	{
		d1.push_back(i);
	}
	printDeque(d1);

	//判断容器是否为空
	if (d1.empty()) {
		cout << "d1为空!" << endl;
	}
	else {
		cout << "d1不为空!" << endl;
		//统计大小
		cout << "d1的大小为：" << d1.size() << endl;
	}

	//重新指定大小
	d1.resize(15, 1);
	printDeque(d1);

	d1.resize(5);
	printDeque(d1);
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

- deque没有容量的概念
- 判断是否为空 — empty
- 返回元素个数 — size
- 重新指定个数 — resize

### deque 插入和删除

**功能描述：**

- 向deque容器中插入和删除数据

**函数原型：**

两端插入操作：

- `push_back(elem);` //在容器尾部添加一个数据
- `push_front(elem);` //在容器头部插入一个数据
- `pop_back();` //删除容器最后一个数据
- `pop_front();` //删除容器第一个数据

指定位置操作：

- `insert(pos,elem);` //在pos位置插入一个elem元素的拷贝，返回新数据的位置。
- `insert(pos,n,elem);` //在pos位置插入n个elem数据，无返回值。
- `insert(pos,beg,end);` //在pos位置插入[beg,end)区间的数据，无返回值。
- `clear();` //清空容器的所有数据
- `erase(beg,end);` //删除[beg,end)区间的数据，返回下一个数据的位置。
- `erase(pos);` //删除pos位置的数据，返回下一个数据的位置。

**示例：**

```c++
C++

#include <deque>

void printDeque(const deque<int>& d) 
{
	for (deque<int>::const_iterator it = d.begin(); it != d.end(); it++) {
		cout << *it << " ";

	}
	cout << endl;
}
//两端操作
void test01()
{
	deque<int> d;
	//尾插
	d.push_back(10);
	d.push_back(20);
	//头插
	d.push_front(100);
	d.push_front(200);

	printDeque(d);

	//尾删
	d.pop_back();
	//头删
	d.pop_front();
	printDeque(d);
}

//插入
void test02()
{
	deque<int> d;
	d.push_back(10);
	d.push_back(20);
	d.push_front(100);
	d.push_front(200);
	printDeque(d);

	d.insert(d.begin(), 1000);
	printDeque(d);

	d.insert(d.begin(), 2,10000);
	printDeque(d);

	deque<int>d2;
	d2.push_back(1);
	d2.push_back(2);
	d2.push_back(3);

	d.insert(d.begin(), d2.begin(), d2.end());
	printDeque(d);

}

//删除
void test03()
{
	deque<int> d;
	d.push_back(10);
	d.push_back(20);
	d.push_front(100);
	d.push_front(200);
	printDeque(d);

	d.erase(d.begin());
	printDeque(d);

	d.erase(d.begin(), d.end());
	d.clear();
	printDeque(d);
}

int main() {

	//test01();

	//test02();

    test03();
    
	system("pause");

	return 0;
}
```

总结：

- 插入和删除提供的位置是迭代器！
- 尾插 — push_back
- 尾删 — pop_back
- 头插 — push_front
- 头删 — pop_front

### deque 数据存取

**功能描述：**

- 对deque 中的数据的存取操作

**函数原型：**

- `at(int idx); `//返回索引idx所指的数据
- `operator[]; `//返回索引idx所指的数据
- `front(); `//返回容器中第一个数据元素
- `back();` //返回容器中最后一个数据元素

**示例：**

```c++
C++

#include <deque>

void printDeque(const deque<int>& d) 
{
	for (deque<int>::const_iterator it = d.begin(); it != d.end(); it++) {
		cout << *it << " ";

	}
	cout << endl;
}

//数据存取
void test01()
{

	deque<int> d;
	d.push_back(10);
	d.push_back(20);
	d.push_front(100);
	d.push_front(200);

	for (int i = 0; i < d.size(); i++) {
		cout << d[i] << " ";
	}
	cout << endl;


	for (int i = 0; i < d.size(); i++) {
		cout << d.at(i) << " ";
	}
	cout << endl;

	cout << "front:" << d.front() << endl;

	cout << "back:" << d.back() << endl;

}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

- 除了用迭代器获取deque容器中元素，[ ]和at也可以
- front返回容器第一个元素
- back返回容器最后一个元素

### deque 排序

**功能描述：**

- 利用算法实现对deque容器进行排序

**算法：**

- `sort(iterator beg, iterator end)` //对beg和end区间内元素进行排序

**示例：**

```c++
C++

#include <deque>
#include <algorithm>

void printDeque(const deque<int>& d) 
{
	for (deque<int>::const_iterator it = d.begin(); it != d.end(); it++) {
		cout << *it << " ";

	}
	cout << endl;
}

void test01()
{

	deque<int> d;
	d.push_back(10);
	d.push_back(20);
	d.push_front(100);
	d.push_front(200);

	printDeque(d);
	sort(d.begin(), d.end());
	printDeque(d);

}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：sort算法非常实用，使用时包含头文件 algorithm即可

## 案例-评委打分

### 案例描述

有5名选手：选手ABCDE，10个评委分别对每一名选手打分，去除最高分，去除评委中最低分，取平均分。

### 实现步骤

1. 创建五名选手，放到vector中
2. 遍历vector容器，取出来每一个选手，执行for循环，可以把10个评分打分存到deque容器中
3. sort算法对deque容器中分数排序，去除最高和最低分
4. deque容器遍历一遍，累加总分
5. 获取平均分

**示例代码：**

```c++
C++

//选手类
class Person
{
public:
	Person(string name, int score)
	{
		this->m_Name = name;
		this->m_Score = score;
	}

	string m_Name; //姓名
	int m_Score;  //平均分
};

void createPerson(vector<Person>&v)
{
	string nameSeed = "ABCDE";
	for (int i = 0; i < 5; i++)
	{
		string name = "选手";
		name += nameSeed[i];

		int score = 0;

		Person p(name, score);

		//将创建的person对象 放入到容器中
		v.push_back(p);
	}
}

//打分
void setScore(vector<Person>&v)
{
	for (vector<Person>::iterator it = v.begin(); it != v.end(); it++)
	{
		//将评委的分数 放入到deque容器中
		deque<int>d;
		for (int i = 0; i < 10; i++)
		{
			int score = rand() % 41 + 60;  // 60 ~ 100
			d.push_back(score);
		}

		//cout << "选手： " << it->m_Name << " 打分： " << endl;
		//for (deque<int>::iterator dit = d.begin(); dit != d.end(); dit++)
		//{
		//	cout << *dit << " ";
		//}
		//cout << endl;

		//排序
		sort(d.begin(), d.end());

		//去除最高和最低分
		d.pop_back();
		d.pop_front();

		//取平均分
		int sum = 0;
		for (deque<int>::iterator dit = d.begin(); dit != d.end(); dit++)
		{
			sum += *dit; //累加每个评委的分数
		}

		int avg = sum / d.size();

		//将平均分 赋值给选手身上
		it->m_Score = avg;
	}

}

void showScore(vector<Person>&v)
{
	for (vector<Person>::iterator it = v.begin(); it != v.end(); it++)
	{
		cout << "姓名： " << it->m_Name << " 平均分： " << it->m_Score << endl;
	}
}

int main() {

	//随机数种子
	srand((unsigned int)time(NULL));

	//1、创建5名选手
	vector<Person>v;  //存放选手容器
	createPerson(v);

	//测试
	//for (vector<Person>::iterator it = v.begin(); it != v.end(); it++)
	//{
	//	cout << "姓名： " << (*it).m_Name << " 分数： " << (*it).m_Score << endl;
	//}

	//2、给5名选手打分
	setScore(v);

	//3、显示最后得分
	showScore(v);

	system("pause");

	return 0;
}
```

**总结：** 选取不同的容器操作数据，可以提升代码的效率

## stack容器

### stack 基本概念

**概念：\**stack是一种\**先进后出**(First In Last Out,FILO)的数据结构，它只有一个出口

[![说明: 2015-11-15_195707](https://pic.imgdb.cn/item/61dd2fc12ab3f51d91787ac4.jpg)](https://pic.imgdb.cn/item/61dd2fc12ab3f51d91787ac4.jpg)

栈中只有顶端的元素才可以被外界使用，因此栈不允许有遍历行为

栈中进入数据称为 — **入栈** `push`

栈中弹出数据称为 — **出栈** `pop`

### stack 常用接口

功能描述：栈容器常用的对外接口

构造函数：

- `stack<T> stk;` //stack采用模板类实现， stack对象的默认构造形式
- `stack(const stack &stk);` //拷贝构造函数

赋值操作：

- `stack& operator=(const stack &stk);` //重载等号操作符

数据存取：

- `push(elem);` //向栈顶添加元素
- `pop();` //从栈顶移除第一个元素
- `top(); `//返回栈顶元素

大小操作：

- `empty();` //判断堆栈是否为空
- `size(); `//返回栈的大小

**示例：**

```c++
C++

#include <stack>

//栈容器常用接口
void test01()
{
	//创建栈容器 栈容器必须符合先进后出
	stack<int> s;

	//向栈中添加元素，叫做 压栈 入栈
	s.push(10);
	s.push(20);
	s.push(30);

	while (!s.empty()) {
		//输出栈顶元素
		cout << "栈顶元素为： " << s.top() << endl;
		//弹出栈顶元素
		s.pop();
	}
	cout << "栈的大小为：" << s.size() << endl;

}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

- 入栈 — push
- 出栈 — pop
- 返回栈顶 — top
- 判断栈是否为空 — empty
- 返回栈大小 — size

## queue 容器

### queue 基本概念

**概念：Queue是一种先进先出**(First In First Out,FIFO)的数据结构，它有两个出口

[![说明: 2015-11-15_214429](https://pic.imgdb.cn/item/61dd30032ab3f51d9178b890.jpg)](https://pic.imgdb.cn/item/61dd30032ab3f51d9178b890.jpg)

队列容器允许从一端新增元素，从另一端移除元素

队列中只有队头和队尾才可以被外界使用，因此队列不允许有遍历行为

队列中进数据称为 — **入队** `push`

队列中出数据称为 — **出队** `pop`

### queue 常用接口

功能描述：栈容器常用的对外接口

构造函数：

- `queue<T> que;` //queue采用模板类实现，queue对象的默认构造形式
- `queue(const queue &que);` //拷贝构造函数

赋值操作：

- `queue& operator=(const queue &que);` //重载等号操作符

数据存取：

- `push(elem);` //往队尾添加元素
- `pop();` //从队头移除第一个元素
- `back();` //返回最后一个元素
- `front(); `//返回第一个元素

大小操作：

- `empty();` //判断堆栈是否为空
- `size(); `//返回栈的大小

**示例：**

```c++
#include <queue>
#include <string>
class Person
{
public:
	Person(string name, int age)
	{
		this->m_Name = name;
		this->m_Age = age;
	}

	string m_Name;
	int m_Age;
};

void test01() {

	//创建队列
	queue<Person> q;

	//准备数据
	Person p1("唐僧", 30);
	Person p2("孙悟空", 1000);
	Person p3("猪八戒", 900);
	Person p4("沙僧", 800);

	//向队列中添加元素  入队操作
	q.push(p1);
	q.push(p2);
	q.push(p3);
	q.push(p4);

	//队列不提供迭代器，更不支持随机访问	
	while (!q.empty()) {
		//输出队头元素
		cout << "队头元素-- 姓名： " << q.front().m_Name 
              << " 年龄： "<< q.front().m_Age << endl;
        
		cout << "队尾元素-- 姓名： " << q.back().m_Name  
              << " 年龄： " << q.back().m_Age << endl;
        
		cout << endl;
		//弹出队头元素
		q.pop();
	}

	cout << "队列大小为：" << q.size() << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

- 入队 — push
- 出队 — pop
- 返回队头元素 — front
- 返回队尾元素 — back
- 判断队是否为空 — empty
- 返回队列大小 — size

## list容器

### list基本概念

**功能：**将数据进行链式存储

**链表**（list）是一种物理存储单元上非连续的存储结构，数据元素的逻辑顺序是通过链表中的指针链接实现的

链表的组成：链表由一系列**结点**组成

结点的组成：一个是存储数据元素的**数据域**，另一个是存储下一个结点地址的**指针域**

STL中的链表是一个双向循环链表

[![说明: 2015-11-15_225145](https://pic.imgdb.cn/item/61dd31062ab3f51d917986eb.jpg)](https://pic.imgdb.cn/item/61dd31062ab3f51d917986eb.jpg)

由于链表的存储方式并不是连续的内存空间，因此链表list中的迭代器只支持前移和后移，属于**双向迭代器**

list的优点：

- 采用动态存储分配，不会造成内存浪费和溢出
- 链表执行插入和删除操作十分方便，修改指针即可，不需要移动大量元素

list的缺点：

- 链表灵活，但是空间(指针域) 和 时间（遍历）额外耗费较大

List有一个重要的性质，插入操作和删除操作都不会造成原有list迭代器的失效，这在vector是不成立的。

总结：STL中**List和vector是两个最常被使用的容器**，各有优缺点

### list构造函数

**功能描述：**

- 创建list容器

**函数原型：**

- `list<T> lst;` //list采用采用模板类实现,对象的默认构造形式：
- `list(beg,end);` //构造函数将[beg, end)区间中的元素拷贝给本身。
- `list(n,elem);` //构造函数将n个elem拷贝给本身。
- `list(const list &lst);` //拷贝构造函数。

**示例：**

```c++
C++

#include <list>

void printList(const list<int>& L) {

	for (list<int>::const_iterator it = L.begin(); it != L.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}

void test01()
{
	list<int>L1;
	L1.push_back(10);
	L1.push_back(20);
	L1.push_back(30);
	L1.push_back(40);

	printList(L1);

	list<int>L2(L1.begin(),L1.end());
	printList(L2);

	list<int>L3(L2);
	printList(L3);

	list<int>L4(10, 1000);
	printList(L4);
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：list构造方式同其他几个STL常用容器，熟练掌握即可

### list 赋值和交换

**功能描述：**

- 给list容器进行赋值，以及交换list容器

**函数原型：**

- `assign(beg, end);` //将[beg, end)区间中的数据拷贝赋值给本身。
- `assign(n, elem);` //将n个elem拷贝赋值给本身。
- `list& operator=(const list &lst);` //重载等号操作符
- `swap(lst);` //将lst与本身的元素互换。

**示例：**

```c++
C++

#include <list>

void printList(const list<int>& L) {

	for (list<int>::const_iterator it = L.begin(); it != L.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}

//赋值和交换
void test01()
{
	list<int>L1;
	L1.push_back(10);
	L1.push_back(20);
	L1.push_back(30);
	L1.push_back(40);
	printList(L1);

	//赋值
	list<int>L2;
	L2 = L1;
	printList(L2);

	list<int>L3;
	L3.assign(L2.begin(), L2.end());
	printList(L3);

	list<int>L4;
	L4.assign(10, 100);
	printList(L4);

}

//交换
void test02()
{

	list<int>L1;
	L1.push_back(10);
	L1.push_back(20);
	L1.push_back(30);
	L1.push_back(40);

	list<int>L2;
	L2.assign(10, 100);

	cout << "交换前： " << endl;
	printList(L1);
	printList(L2);

	cout << endl;

	L1.swap(L2);

	cout << "交换后： " << endl;
	printList(L1);
	printList(L2);

}

int main() {

	//test01();

	test02();

	system("pause");

	return 0;
}
```

总结：list赋值和交换操作能够灵活运用即可

### list 大小操作

**功能描述：**

- 对list容器的大小进行操作

**函数原型：**

- `size(); `//返回容器中元素的个数

- `empty(); `//判断容器是否为空

- `resize(num);` //重新指定容器的长度为num，若容器变长，则以默认值填充新位置。

	//如果容器变短，则末尾超出容器长度的元素被删除。

- `resize(num, elem); `//重新指定容器的长度为num，若容器变长，则以elem值填充新位置。

	```
	  	 	 						    //如果容器变短，则末尾超出容器长度的元素被删除。
	```

**示例：**

```c++
C++

#include <list>

void printList(const list<int>& L) {

	for (list<int>::const_iterator it = L.begin(); it != L.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}

//大小操作
void test01()
{
	list<int>L1;
	L1.push_back(10);
	L1.push_back(20);
	L1.push_back(30);
	L1.push_back(40);

	if (L1.empty())
	{
		cout << "L1为空" << endl;
	}
	else
	{
		cout << "L1不为空" << endl;
		cout << "L1的大小为： " << L1.size() << endl;
	}

	//重新指定大小
	L1.resize(10);
	printList(L1);

	L1.resize(2);
	printList(L1);
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

- 判断是否为空 — empty
- 返回元素个数 — size
- 重新指定个数 — resize

### list 插入和删除

**功能描述：**

- 对list容器进行数据的插入和删除

**函数原型：**

- push_back(elem);//在容器尾部加入一个元素
- pop_back();//删除容器中最后一个元素
- push_front(elem);//在容器开头插入一个元素
- pop_front();//从容器开头移除第一个元素
- insert(pos,elem);//在pos位置插elem元素的拷贝，返回新数据的位置。
- insert(pos,n,elem);//在pos位置插入n个elem数据，无返回值。
- insert(pos,beg,end);//在pos位置插入[beg,end)区间的数据，无返回值。
- clear();//移除容器的所有数据
- erase(beg,end);//删除[beg,end)区间的数据，返回下一个数据的位置。
- erase(pos);//删除pos位置的数据，返回下一个数据的位置。
- remove(elem);//删除容器中所有与elem值匹配的元素。

**示例：**

```c++
C++

#include <list>

void printList(const list<int>& L) {

	for (list<int>::const_iterator it = L.begin(); it != L.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}

//插入和删除
void test01()
{
	list<int> L;
	//尾插
	L.push_back(10);
	L.push_back(20);
	L.push_back(30);
	//头插
	L.push_front(100);
	L.push_front(200);
	L.push_front(300);

	printList(L);

	//尾删
	L.pop_back();
	printList(L);

	//头删
	L.pop_front();
	printList(L);

	//插入
	list<int>::iterator it = L.begin();
	L.insert(++it, 1000);
	printList(L);

	//删除
	it = L.begin();
	L.erase(++it);
	printList(L);

	//移除
	L.push_back(10000);
	L.push_back(10000);
	L.push_back(10000);
	printList(L);
	L.remove(10000);
	printList(L);
    
    //清空
	L.clear();
	printList(L);
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

- 尾插 — push_back
- 尾删 — pop_back
- 头插 — push_front
- 头删 — pop_front
- 插入 — insert
- 删除 — erase
- 移除 — remove
- 清空 — clear

### list 数据存取

**功能描述：**

- 对list容器中数据进行存取

**函数原型：**

- `front();` //返回第一个元素。
- `back();` //返回最后一个元素。

**示例：**

```c++
C++

#include <list>

//数据存取
void test01()
{
	list<int>L1;
	L1.push_back(10);
	L1.push_back(20);
	L1.push_back(30);
	L1.push_back(40);

	
	//cout << L1.at(0) << endl;//错误 不支持at访问数据
	//cout << L1[0] << endl; //错误  不支持[]方式访问数据
	cout << "第一个元素为： " << L1.front() << endl;
	cout << "最后一个元素为： " << L1.back() << endl;

	//list容器的迭代器是双向迭代器，不支持随机访问
	list<int>::iterator it = L1.begin();
	//it = it + 1;//错误，不可以跳跃访问，即使是+1
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

- list容器中不可以通过[]或者at方式访问数据
- 返回第一个元素 — front
- 返回最后一个元素 — back

### list 反转和排序

**功能描述：**

- 将容器中的元素反转，以及将容器中的数据进行排序

**函数原型：**

- `reverse();` //反转链表
- `sort();` //链表排序

**示例：**

```c++
C++

void printList(const list<int>& L) {

	for (list<int>::const_iterator it = L.begin(); it != L.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}

bool myCompare(int val1 , int val2)
{
	return val1 > val2;
}

//反转和排序
void test01()
{
	list<int> L;
	L.push_back(90);
	L.push_back(30);
	L.push_back(20);
	L.push_back(70);
	printList(L);

	//反转容器的元素
	L.reverse();
	printList(L);

	//排序
	L.sort(); //默认的排序规则 从小到大
	printList(L);

	L.sort(myCompare); //指定规则，从大到小
	printList(L);
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

- 反转 — reverse
- 排序 — sort （成员函数）

### 排序案例

案例描述：将Person自定义数据类型进行排序，Person中属性有姓名、年龄、身高

排序规则：按照年龄进行升序，如果年龄相同按照身高进行降序

**示例：**

```c++
C++

#include <list>
#include <string>
class Person {
public:
	Person(string name, int age , int height) {
		m_Name = name;
		m_Age = age;
		m_Height = height;
	}

public:
	string m_Name;  //姓名
	int m_Age;      //年龄
	int m_Height;   //身高
};


bool ComparePerson(Person& p1, Person& p2) {

	if (p1.m_Age == p2.m_Age) {
		return p1.m_Height  > p2.m_Height;
	}
	else
	{
		return  p1.m_Age < p2.m_Age;
	}

}

void test01() {

	list<Person> L;

	Person p1("刘备", 35 , 175);
	Person p2("曹操", 45 , 180);
	Person p3("孙权", 40 , 170);
	Person p4("赵云", 25 , 190);
	Person p5("张飞", 35 , 160);
	Person p6("关羽", 35 , 200);

	L.push_back(p1);
	L.push_back(p2);
	L.push_back(p3);
	L.push_back(p4);
	L.push_back(p5);
	L.push_back(p6);

	for (list<Person>::iterator it = L.begin(); it != L.end(); it++) {
		cout << "姓名： " << it->m_Name << " 年龄： " << it->m_Age 
              << " 身高： " << it->m_Height << endl;
	}

	cout << "---------------------------------" << endl;
	L.sort(ComparePerson); //排序

	for (list<Person>::iterator it = L.begin(); it != L.end(); it++) {
		cout << "姓名： " << it->m_Name << " 年龄： " << it->m_Age 
              << " 身高： " << it->m_Height << endl;
	}
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

- 对于自定义数据类型，必须要指定排序规则，否则编译器不知道如何进行排序
- 高级排序只是在排序规则上再进行一次逻辑规则制定，并不复杂

## set/ multiset 容器

### 3.8.1 set基本概念

**简介：**

- 所有元素都会在插入时自动被排序

**本质：**

- set/multiset属于**关联式容器**，底层结构是用**二叉树**实现。

**set和multiset区别**：

- set不允许容器中有重复的元素
- multiset允许容器中有重复的元素

### set构造和赋值

功能描述：创建set容器以及赋值

构造：

- `set<T> st;` //默认构造函数：
- `set(const set &st);` //拷贝构造函数

赋值：

- `set& operator=(const set &st);` //重载等号操作符

**示例：**

```c++
C++

#include <set>

void printSet(set<int> & s)
{
	for (set<int>::iterator it = s.begin(); it != s.end(); it++)
	{
		cout << *it << " ";
	}
	cout << endl;
}

//构造和赋值
void test01()
{
	set<int> s1;

	s1.insert(10);
	s1.insert(30);
	s1.insert(20);
	s1.insert(40);
	printSet(s1);

	//拷贝构造
	set<int>s2(s1);
	printSet(s2);

	//赋值
	set<int>s3;
	s3 = s2;
	printSet(s3);
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

- set容器插入数据时用insert
- set容器插入数据的数据会自动排序

### set大小和交换

**功能描述：**

- 统计set容器大小以及交换set容器

**函数原型：**

- `size();` //返回容器中元素的数目
- `empty();` //判断容器是否为空
- `swap(st);` //交换两个集合容器

**示例：**

```c++
C++

#include <set>

void printSet(set<int> & s)
{
	for (set<int>::iterator it = s.begin(); it != s.end(); it++)
	{
		cout << *it << " ";
	}
	cout << endl;
}

//大小
void test01()
{

	set<int> s1;
	
	s1.insert(10);
	s1.insert(30);
	s1.insert(20);
	s1.insert(40);

	if (s1.empty())
	{
		cout << "s1为空" << endl;
	}
	else
	{
		cout << "s1不为空" << endl;
		cout << "s1的大小为： " << s1.size() << endl;
	}

}

//交换
void test02()
{
	set<int> s1;

	s1.insert(10);
	s1.insert(30);
	s1.insert(20);
	s1.insert(40);

	set<int> s2;

	s2.insert(100);
	s2.insert(300);
	s2.insert(200);
	s2.insert(400);

	cout << "交换前" << endl;
	printSet(s1);
	printSet(s2);
	cout << endl;

	cout << "交换后" << endl;
	s1.swap(s2);
	printSet(s1);
	printSet(s2);
}

int main() {

	//test01();

	test02();

	system("pause");

	return 0;
}
```

总结：

- 统计大小 — size
- 判断是否为空 — empty
- 交换容器 — swap

### set插入和删除

**功能描述：**

- set容器进行插入数据和删除数据

**函数原型：**

- `insert(elem);` //在容器中插入元素。
- `clear();` //清除所有元素
- `erase(pos);` //删除pos迭代器所指的元素，返回下一个元素的迭代器。
- `erase(beg, end);` //删除区间[beg,end)的所有元素 ，返回下一个元素的迭代器。
- `erase(elem);` //删除容器中值为elem的元素。

**示例：**

```c++
C++

#include <set>

void printSet(set<int> & s)
{
	for (set<int>::iterator it = s.begin(); it != s.end(); it++)
	{
		cout << *it << " ";
	}
	cout << endl;
}

//插入和删除
void test01()
{
	set<int> s1;
	//插入
	s1.insert(10);
	s1.insert(30);
	s1.insert(20);
	s1.insert(40);
	printSet(s1);

	//删除
	s1.erase(s1.begin());
	printSet(s1);

	s1.erase(30);
	printSet(s1);

	//清空
	//s1.erase(s1.begin(), s1.end());
	s1.clear();
	printSet(s1);
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

- 插入 — insert
- 删除 — erase
- 清空 — clear

### set查找和统计

**功能描述：**

- 对set容器进行查找数据以及统计数据

**函数原型：**

- `find(key);` //查找key是否存在,若存在，返回该键的元素的迭代器；若不存在，返回set.end();
- `count(key);` //统计key的元素个数

**示例：**

```c++
C++

#include <set>

//查找和统计
void test01()
{
	set<int> s1;
	//插入
	s1.insert(10);
	s1.insert(30);
	s1.insert(20);
	s1.insert(40);
	
	//查找
	set<int>::iterator pos = s1.find(30);

	if (pos != s1.end())
	{
		cout << "找到了元素 ： " << *pos << endl;
	}
	else
	{
		cout << "未找到元素" << endl;
	}

	//统计
	int num = s1.count(30);
	cout << "num = " << num << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

- 查找 — find （返回的是迭代器）
- 统计 — count （对于set，结果为0或者1）

### set和multiset区别

**学习目标：**

- 掌握set和multiset的区别

**区别：**

- set不可以插入重复数据，而multiset可以
- set插入数据的同时会返回插入结果，表示插入是否成功
- multiset不会检测数据，因此可以插入重复数据

**示例：**

```c++
C++

#include <set>

//set和multiset区别
void test01()
{
	set<int> s;
	pair<set<int>::iterator, bool>  ret = s.insert(10);
	if (ret.second) {
		cout << "第一次插入成功!" << endl;
	}
	else {
		cout << "第一次插入失败!" << endl;
	}

	ret = s.insert(10);
	if (ret.second) {
		cout << "第二次插入成功!" << endl;
	}
	else {
		cout << "第二次插入失败!" << endl;
	}
    
	//multiset
	multiset<int> ms;
	ms.insert(10);
	ms.insert(10);

	for (multiset<int>::iterator it = ms.begin(); it != ms.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

- 如果不允许插入重复数据可以利用set
- 如果需要插入重复数据利用multiset

###  pair对组创建

**功能描述：**

- 成对出现的数据，利用对组可以返回两个数据

**两种创建方式：**

- `pair<type, type> p ( value1, value2 );`
- `pair<type, type> p = make_pair( value1, value2 );`

**示例：**

```c++
C++

#include <string>

//对组创建
void test01()
{
	pair<string, int> p(string("Tom"), 20);
	cout << "姓名： " <<  p.first << " 年龄： " << p.second << endl;

	pair<string, int> p2 = make_pair("Jerry", 10);
	cout << "姓名： " << p2.first << " 年龄： " << p2.second << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

两种方式都可以创建对组，记住一种即可

### set容器排序

学习目标：

- set容器默认排序规则为从小到大，掌握如何改变排序规则

主要技术点：

- 利用仿函数，可以改变排序规则

**示例一** set存放内置数据类型

```c++
C++

#include <set>

class MyCompare 
{
public:
	bool operator()(int v1, int v2) {
		return v1 > v2;
	}
};
void test01() 
{    
	set<int> s1;
	s1.insert(10);
	s1.insert(40);
	s1.insert(20);
	s1.insert(30);
	s1.insert(50);

	//默认从小到大
	for (set<int>::iterator it = s1.begin(); it != s1.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;

	//指定排序规则
	set<int,MyCompare> s2;
	s2.insert(10);
	s2.insert(40);
	s2.insert(20);
	s2.insert(30);
	s2.insert(50);

	for (set<int, MyCompare>::iterator it = s2.begin(); it != s2.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：利用仿函数可以指定set容器的排序规则

**示例二** set存放自定义数据类型

```c++
C++

#include <set>
#include <string>

class Person
{
public:
	Person(string name, int age)
	{
		this->m_Name = name;
		this->m_Age = age;
	}

	string m_Name;
	int m_Age;

};
class comparePerson
{
public:
	bool operator()(const Person& p1, const Person &p2)
	{
		//按照年龄进行排序  降序
		return p1.m_Age > p2.m_Age;
	}
};

void test01()
{
	set<Person, comparePerson> s;

	Person p1("刘备", 23);
	Person p2("关羽", 27);
	Person p3("张飞", 25);
	Person p4("赵云", 21);

	s.insert(p1);
	s.insert(p2);
	s.insert(p3);
	s.insert(p4);

	for (set<Person, comparePerson>::iterator it = s.begin(); it != s.end(); it++)
	{
		cout << "姓名： " << it->m_Name << " 年龄： " << it->m_Age << endl;
	}
}
int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

对于自定义数据类型，set必须指定排序规则才可以插入数据

## map/ multimap容器

### map基本概念

**简介：**

- map中所有元素都是pair
- pair中第一个元素为key（键值），起到索引作用，第二个元素为value（实值）
- 所有元素都会根据元素的键值自动排序

**本质：**

- map/multimap属于**关联式容器**，底层结构是用二叉树实现。

**优点：**

- 可以根据key值快速找到value值

map和multimap**区别**：

- map不允许容器中有重复key值元素
- multimap允许容器中有重复key值元素

### map构造和赋值

**功能描述：**

- 对map容器进行构造和赋值操作

**函数原型：**

**构造：**

- `map<T1, T2> mp;` //map默认构造函数:
- `map(const map &mp);` //拷贝构造函数

**赋值：**

- `map& operator=(const map &mp);` //重载等号操作符

**示例：**

```c++
C++

#include <map>

void printMap(map<int,int>&m)
{
	for (map<int, int>::iterator it = m.begin(); it != m.end(); it++)
	{
		cout << "key = " << it->first << " value = " << it->second << endl;
	}
	cout << endl;
}

void test01()
{
	map<int,int>m; //默认构造
	m.insert(pair<int, int>(1, 10));
	m.insert(pair<int, int>(2, 20));
	m.insert(pair<int, int>(3, 30));
	printMap(m);

	map<int, int>m2(m); //拷贝构造
	printMap(m2);

	map<int, int>m3;
	m3 = m2; //赋值
	printMap(m3);
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：map中所有元素都是成对出现，插入数据时候要使用对组

### map大小和交换

**功能描述：**

- 统计map容器大小以及交换map容器

函数原型：

- `size();` //返回容器中元素的数目
- `empty();` //判断容器是否为空
- `swap(st);` //交换两个集合容器

**示例：**

```c++
C++

#include <map>

void printMap(map<int,int>&m)
{
	for (map<int, int>::iterator it = m.begin(); it != m.end(); it++)
	{
		cout << "key = " << it->first << " value = " << it->second << endl;
	}
	cout << endl;
}

void test01()
{
	map<int, int>m;
	m.insert(pair<int, int>(1, 10));
	m.insert(pair<int, int>(2, 20));
	m.insert(pair<int, int>(3, 30));

	if (m.empty())
	{
		cout << "m为空" << endl;
	}
	else
	{
		cout << "m不为空" << endl;
		cout << "m的大小为： " << m.size() << endl;
	}
}


//交换
void test02()
{
	map<int, int>m;
	m.insert(pair<int, int>(1, 10));
	m.insert(pair<int, int>(2, 20));
	m.insert(pair<int, int>(3, 30));

	map<int, int>m2;
	m2.insert(pair<int, int>(4, 100));
	m2.insert(pair<int, int>(5, 200));
	m2.insert(pair<int, int>(6, 300));

	cout << "交换前" << endl;
	printMap(m);
	printMap(m2);

	cout << "交换后" << endl;
	m.swap(m2);
	printMap(m);
	printMap(m2);
}

int main() {

	test01();

	test02();

	system("pause");

	return 0;
}
```

总结：

- 统计大小 — size
- 判断是否为空 — empty
- 交换容器 — swap

### map插入和删除

**功能描述：**

- map容器进行插入数据和删除数据

**函数原型：**

- `insert(elem);` //在容器中插入元素。
- `clear();` //清除所有元素
- `erase(pos);` //删除pos迭代器所指的元素，返回下一个元素的迭代器。
- `erase(beg, end);` //删除区间[beg,end)的所有元素 ，返回下一个元素的迭代器。
- `erase(key);` //删除容器中值为key的元素。

**示例：**

```c++
C++

#include <map>

void printMap(map<int,int>&m)
{
	for (map<int, int>::iterator it = m.begin(); it != m.end(); it++)
	{
		cout << "key = " << it->first << " value = " << it->second << endl;
	}
	cout << endl;
}

void test01()
{
	//插入
	map<int, int> m;
	//第一种插入方式
	m.insert(pair<int, int>(1, 10));
	//第二种插入方式
	m.insert(make_pair(2, 20));
	//第三种插入方式
	m.insert(map<int, int>::value_type(3, 30));
	//第四种插入方式
	m[4] = 40; 
	printMap(m);

	//删除
	m.erase(m.begin());
	printMap(m);

	m.erase(3);
	printMap(m);

	//清空
	m.erase(m.begin(),m.end());
	m.clear();
	printMap(m);
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

- map插入方式很多，记住其一即可

- 插入 — insert
- 删除 — erase
- 清空 — clear

### map查找和统计

**功能描述：**

- 对map容器进行查找数据以及统计数据

**函数原型：**

- `find(key);` //查找key是否存在,若存在，返回该键的元素的迭代器；若不存在，返回set.end();
- `count(key);` //统计key的元素个数

**示例：**

```c++
C++

#include <map>

//查找和统计
void test01()
{
	map<int, int>m; 
	m.insert(pair<int, int>(1, 10));
	m.insert(pair<int, int>(2, 20));
	m.insert(pair<int, int>(3, 30));

	//查找
	map<int, int>::iterator pos = m.find(3);

	if (pos != m.end())
	{
		cout << "找到了元素 key = " << (*pos).first << " value = " << (*pos).second << endl;
	}
	else
	{
		cout << "未找到元素" << endl;
	}

	//统计
	int num = m.count(3);
	cout << "num = " << num << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

- 查找 — find （返回的是迭代器）
- 统计 — count （对于map，结果为0或者1）

### map容器排序

**学习目标：**

- map容器默认排序规则为 按照key值进行 从小到大排序，掌握如何改变排序规则

**主要技术点:**

- 利用仿函数，可以改变排序规则

**示例：**

```c++
C++

#include <map>

class MyCompare {
public:
	bool operator()(int v1, int v2) {
		return v1 > v2;
	}
};

void test01() 
{
	//默认从小到大排序
	//利用仿函数实现从大到小排序
	map<int, int, MyCompare> m;

	m.insert(make_pair(1, 10));
	m.insert(make_pair(2, 20));
	m.insert(make_pair(3, 30));
	m.insert(make_pair(4, 40));
	m.insert(make_pair(5, 50));

	for (map<int, int, MyCompare>::iterator it = m.begin(); it != m.end(); it++) {
		cout << "key:" << it->first << " value:" << it->second << endl;
	}
}
int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：

- 利用仿函数可以指定map容器的排序规则
- 对于自定义数据类型，map必须要指定排序规则,同set容器

## priority_queue容器

### 基本概念

底层基于完全二叉树

常用方法：

```c++
q.size();//返回q里元素个数
q.empty();//返回q是否为空，空则返回1，否则返回0
q.push(k);//在q的末尾插入k
q.pop();//删掉q的第一个元素,使用STL是删除堆顶，手写堆可以自行决定
q.top();//返回q的第一个元素
```

默认的优先队列（非结构体结构）:
```c++
priority_queue <int> p;//从大到小排序
```

使用重载结构:

```c++
//less<int> >q中右边是分开的'> >'合并就是右移运算符
priority_queue<int, vector<int>, less<int> >p;//从大到小排序
priority_queue<int, vector<int>, greater<int> >q;//从小到大排序
```

# 常见库函数方法

## 数据类型转换

```c++
//to_string（转换字符串
功能：将数字常量转换为字符串
int num = 123456789;
cout << to_string(num) << endl;//"123456789"



// stoi（转换十进制
功能：将n进制的字符串转换为十进制
stoi（字符串，起始位置，n进制（默认10进制）），将 n 进制的字符串转化为十进制 
string str = "100";
int x = stoi(str, 0, 2); //将二进制"100"转化为十进制x
cout << x << endl;//4



//bitset（位操作
功能：适合表示二进制数据或进行位操作
1.bitset<4> a； //申请一个名为a长度为4的bitset，默认每位为0
2.bitset<8> b(12)； //长度为8，将12的二进制保存在b中，前面位补0
3.string s = "10010";
bitset<10> c(s);  //长度为10，将s的二进制保存在c中，前面位补0

cout << a << endl; //0000
cout << b << endl; //00001100
cout << c << endl; //0000010010



// isdigit（是否十进制数字字符
功能：检查字符是否是十进制数字字符
int ch1 = 'h';
int ch2 = '2';

cout << isalnum(ch1);//false
cout << isalnum(ch2);//true




//isalnum（是否数字或字母
功能：检查字符是否是数字或者字母
char ch1 = 'a';
char ch2 = '2';
char ch3 = '.';

cout << isalnum(ch1);//true
cout << isalnum(ch2);//true
cout << isalnum(ch3);//fasle



//isalpha（是否字母
功能：检查字符是否是字母
char ch1 = 'A'; // 大写字母
char ch2 = 'z'; // 小写字母
char ch3 = '5'; // 数字

cout << isalnum(ch1);//true
cout << isalnum(ch2);//true
cout << isalnum(ch3);//false



// isupper（是否字母大写
功能：检查字母是否大写
char ch1 = 'A'; // 大写字母
char ch2 = 'a'; // 小写字母
char ch3 = '5'; // 数字

cout << isalnum(ch1);//true
cout << isalnum(ch2);//false
cout << isalnum(ch3);//fasle



// islower（是否字母小写
功能：检查字母是否小写
char ch1 = 'A'; // 大写字母
char ch2 = 'a'; // 小写字母
char ch3 = '5'; // 数字

cout << isalnum(ch1);//false
cout << isalnum(ch2);//true
cout << isalnum(ch3);//fasle



// isascii（是否ASCII
功能：检查字符是否是ASCII码
char ch1 = 'c';

cout << isascii(ch1);//true



// toupper（转换字符为大写
功能：将字母转换为大写
char ch1 = 'a'; // 小写字母

cout << toupper(ch1);//A



// tolower（转换字符为小写
功能：将字母转换为小写字母
char ch1 = 'A'; // 大写字母

cout << tolower(ch1);//a



// toascii（转换为ASCII
功能：将字符转换为ASCII码
```

### toBinary（自定义十进制转二进制函数

功能：自定义十进制转二进制函数

```c++
string toBinary(int num){
    string res;
    while(num != 0){
        res = (num % 2 == 0 ? "0":"1") + res;
        num /= 2;
    }
    return res;
}
```



## 数学类型

```c++
float fabs(float x)	求浮点数x的绝对值
int abs(int x)	求整数x的绝对值
float pow(float x, float y)	计算x的y次幂
float pow10(float x)	计算10的x次幂
float sqrt(float x)	计算x的平方根
```



## lower_bound（查找大于等于关键字的元素

功能：在有序序列里查找**大于等于**关键字的元素，并返回元素索引位置最低的地址，最后根据地址来判断是否查找成功

```c++
lower_bound( begin , end , val)
```

**自定义比较函数：**

**less<type>()** 自定义比较函数：适用于**从小到大排序**的有序序列，从数组/容器的 **beign** 位置起，到 **end-1** 位置结束，查找**第一个大于等于 **val 的数字

```c++
lower_bound( begin , end , val , less<type>() )
```

**greater<type>()** 自定义比较函数：适用于**从大到小排序**的有序序列，从数组/容器的 **beign** 位置起，到 **end-1** 位置结束，查找**第一个 小于等于** val 的数字

```c++
lower_bound( begin , end , val , greater<type>() )
```



## upper_bound（查找大于关键字的元素

功能：在有序序列里查找**大于**关键字的元素，并返回元素索引位置最低的地址，最后根据地址来判断是否查找成功

```c++
upper_bound( begin , end , val)
```

**less<type>()** 自定义比较函数：适用于从小到大排序的有序序列，从数组/容器的 beign 位置起，到 end-1 位置结束，查找**第一个 大于** val 的数字

```c++
upper_bound( begin , end , val , less<type>() )
```


 **greater<type>()** 自定义比较函数：适用于从大到小排序的有序序列，从数组/容器的 beign 位置起，到 end-1 位置结束，查找**第一个 小于** val 的数字

```c++
upper_bound( begin , end , val , greater<type>() )
```



## reverse（ 翻转 

功能：翻转数组，字符串，向量

```c++
string str = "abcd";
int arr[5] = {5,4,3,2,1};

cout << reverse(str.begin(), str.end());//"dcba"
cout << reverse(arr.begin(), arr.end());//1,2,3,4,5
```

