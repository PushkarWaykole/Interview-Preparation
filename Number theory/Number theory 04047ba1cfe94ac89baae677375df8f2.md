# Number theory

Modular Arithmetic

![Untitled](Number%20theory%2004047ba1cfe94ac89baae677375df8f2/Untitled.png)

M=10^9+7 ⇒very close to max value of integer and it is prime number

The last bit of odd number is always 1

![Untitled](Number%20theory%2004047ba1cfe94ac89baae677375df8f2/Untitled%201.png)

3<<1 means shift 11 to left ⇒ 110

For n bits we can store maximum (2^n)-1 numbers

1<<n means 2^n

signed int=→  1 bit is reserved for the sign of the number ⇒ max 2^31-1

unsigned int ⇒ 2^32-1

Most significant bit is the first bit in the binary number

Least significant bit is the leftmost bit of the binary number

To check if the ith bit is set or not

**if(num &(1<<i) ≠0) means the bit is set**

To set the ith bit  ⇒ **num | (1<<i)**

To unset the ith bit ⇒ **num & ~(1<<i)**

> ~ means all the set bits are unset and all the unsets bits are set
> 

Toggle bit

**num ^ (1<<i)**

To count number of set bits

**__builtin_popcount(4)**

For long long int  ⇒ __builtin_popcountll(4)

To check odd even 

if (n&1) odd

else even

To convert upper case to lower case ⇒ ‘A’ | “” ⇒ a

To convert lowercase to upper case ⇒ “a” & “_” ⇒ A

Check if the number is power of 2 ⇒ (n & (n-1))==0

```cpp
//Swap two numbers using Xor
int a=4,b=6;

a=a^b;

b=b^a;  // b=b^(a^b) => b=a

a=a^b; // a=(a^b) ^ a => b

The int a and b are swaped

```

```cpp
//Given an array all integers are present in even count except one 
//Find that number

//Example 2 4 6 7 7 4 2 2 2 => output => 6

Just take the XOR of all the elements of the array
the ans will be the numjber which is present only once

```

## Bit masking

We can find intersection of 2 sorted intersection in O(N) time complexity using two pointers

### Suppose there are 3 people and all of them have any number of the given 4 fruits

Apple→0

Orange→1

Banana→2

Litchi→3

Person 1→ 2,3   using bitmask we can represent the number in which the 2nd and 3rd bit is set ⇒ 1100 ⇒ 12

Person 2→ 0,1,2 ⇒ 0111⇒7

Person 3 → 1,3 ⇒ 1010⇒10

Now find the common fruits in 1 and 2 person

We can take AND of 1100 and 0111 ⇒ 0100 ⇒ only 1 fruits is common

unsigned long long ⇒ 64 bit that means we can use bitmask only if the number of items are less than 64

```cpp
//Consider this problem: There are N≤5000 workers. Each worker is available during some days of this month (which has 30 days). For each worker, you are given a set of numbers, each from interval [1,30], representing his/her availability. You need to assign an important project to two workers but they will be able to work on the project only when they are both available. Find two workers that are best for the job — maximize the number of days when both these workers are available.

vector<int> masks(n, 0);
for (int i = 0; i < n; i++)
{
    int num_workers;
    cin >> num_workers;
    int mask = 0;
    for (int j = 0; j < n; j++)
    {
        int day;
        cin >> day;
        mask = (mask | (1 << day));
    }
    masks[i] = mask;
}

int ans = 0;
int p1 = -1, p2 = -1;
for (int i = 0; i < n; i++)
{
    for (int j = i + 1; j < n; j++)
    {
        int intersection = masks[i] & masks[j];
        int common_days = __builtin_popcount(intersection);
        if (ans > common_days)
        {
            ans = common;
            p1 = i;
            p2 = j;
        }
    }
}

```

The inbuilt pow() function returns a double and double is not precise 

Binary Exponentiation recursive code

```cpp
int solve(int a,int b){
	if(b==0) return 1;
	long long res=solve(a/b/2);
if(b&1){
	return a*res*res;
}
else{
	return res*res;
}

}
```

## Iterative version of binary exponentiation

```cpp
int m=1e9+7;
int solve(int a,int b){
	int ans=1;
	while(b){
		if(b&1){
			ans=(ans*a)%m;
		}
		a=(a*a)%m;
		b>>=1;	
	}
	return ans;

}
```

## If the value of b is large ie b≤ 1e18

 

**Two numbers are co prime if gcd(a,b)=1**

ETF ⇒ Euler totient function

For N ⇒ count of k such that 1≤k≤N , k and N are coprime

Eg → 5⇒ 1,2,3,4     

5 and 5 are not co prime

![Untitled](Number%20theory%2004047ba1cfe94ac89baae677375df8f2/Untitled%202.png)

$$
fi(5)=4
$$

$fi(n)=n\cdot Multiplication(1- 1/p)$

if n is prime

**then fi(n)=n-1**

 Eulers theorem

![Untitled](Number%20theory%2004047ba1cfe94ac89baae677375df8f2/Untitled%203.png)

![Untitled](Number%20theory%2004047ba1cfe94ac89baae677375df8f2/Untitled%204.png)

# Divisors

Brute force approach

```cpp
		// O(n)

		int n;
    cin>>n;
    for(int i=1;i<n;i++){
        if(n%i==0){
            cout<<i<<endl;
        }
    }
```

```cpp
		
iterating till root only => O(sqrt(n))

int n;
    cin>>n;
	int count=0,sum=0;
    for(int i=1;i*i<n;i++){
        if(n%i==0){
            cout<<i<" "<<n/i<<endl;
						count+=1;
						sum+=i;
						if(n/i!=i){
						sum+=i;
						count+=1
						}
        }
    }
```

Formula for number of divisors of a number

![Untitled](Number%20theory%2004047ba1cfe94ac89baae677375df8f2/Untitled%205.png)

Sum of divisors

![Untitled](Number%20theory%2004047ba1cfe94ac89baae677375df8f2/Untitled%206.png)

  

## Code to find the number of divisors and sum of those divisors in O(sqrt(n)) time complexity

```cpp
				int n;
        cin>>n;
        int sum=0,number=0;
        if(n==1){
            print(0);
        }
        else{
            for(int i=2;i<=sqrt(n);i++){
                if(n%i==0){
                    if(i==(n/i)){
                        sum+=i;
                        number+=1;
                        
                    }
                    else{
                        sum+=(i+n/i);
                        number+=2;
                    }
                }
            }
            cout<<"The sum of divisors of "<<n<<" is "<<sum+1<<endl;
            cout<<"The number of divisors of "<<n<<" is "<<number<<endl;
        }
```

# Prime numbers

Brute force method to find if the number is prime or not

![Untitled](Number%20theory%2004047ba1cfe94ac89baae677375df8f2/Untitled%207.png)

Now we need to check till sqrt(N) only

> The smallest divisor of any number is always **prime excluding one**
> 

## Printing the prime factors of a number

![Untitled](Number%20theory%2004047ba1cfe94ac89baae677375df8f2/Untitled%208.png)

> For non prime numbers ,there is always a prime number before sqrt(n)
> 

O(sqrt(N)) time complexity

![Untitled](Number%20theory%2004047ba1cfe94ac89baae677375df8f2/Untitled%209.png)

# Seive algorithm

There are Q queries and N

Q<10^7,N<10^7

Find if the number is prime or not

 **O(sqrt(N)) will give TLE**

![Untitled](Number%20theory%2004047ba1cfe94ac89baae677375df8f2/Untitled%2010.png)

**Time complexity - O(log log(N))**

Optimized code

```cpp
vector<ll> sieve(n+1,0);
        
        
        
        for (ll x = 2; x*x <= n; x++) {
		    if (sieve[x]) continue;
		    for (ll u = x*x; u <= n; u += x) {
					sieve[u] = 1;
		    }
        }
```

### To find the highest and lowest prime of a number

Example for 10 → 2*5

highest prime=5

lowest prime=2

![Untitled](Number%20theory%2004047ba1cfe94ac89baae677375df8f2/Untitled%2011.png)

lp is lowest prime and hp is the highest prime

Compute the prime factors

![Untitled](Number%20theory%2004047ba1cfe94ac89baae677375df8f2/Untitled%2012.png)

### To store divisors of all numbers upto N

![Untitled](Number%20theory%2004047ba1cfe94ac89baae677375df8f2/Untitled%2013.png)

# Multiplicative inverse → O(m)

![Untitled](Number%20theory%2004047ba1cfe94ac89baae677375df8f2/Untitled%2014.png)

![Untitled](Number%20theory%2004047ba1cfe94ac89baae677375df8f2/Untitled%2015.png)

**Multiplicative inverse exist if gcd(A,M)=1 that is they are coprime**

**This is Fermet Little theorem**

which means when we divide A^(M-1) by M we get the reminder as 1

This expression is congruency of numbers

![Untitled](Number%20theory%2004047ba1cfe94ac89baae677375df8f2/Untitled%2016.png)

The below is the multiplicative inverse of A with M

That is binary_exp(A,M-2)

![Untitled](Number%20theory%2004047ba1cfe94ac89baae677375df8f2/Untitled%2017.png)

## Question on modular arithmetic

**1680. Concatenation of Consecutive Binary Numbers**

Given an integer `n`, return *the **decimal value** of the binary string formed by concatenating the binary representations of* `1` *to* `n` *in order, **modulo*** `109 + 7`.

**Example 1:**

```
Input: n = 1
Output: 1
Explanation:"1" in binary corresponds to the decimal value 1.

```

**Example 2:**

```
Input: n = 3
Output: 27
Explanation:In binary, 1, 2, and 3 corresponds to "1", "10", and "11".
After concatenating them, we have "11011", which corresponds to the decimal value 27.

```

**Example 3:**

```
Input: n = 12
Output: 505379714
Explanation: The concatenation results in "1101110010111011110001001101010111100".
The decimal value of that is 118505380540.
After modulo 109 + 7, the result is 505379714.

```

**Constraints:**

- `1 <= n <= 105`

```cpp
class Solution {
public:
    int numberOfBits(int n) {
		  return log2(n) + 1;
    }
    
    int concatenatedBinary(int n) {
        long ans = 0, MOD = 1e9 + 7;
        
        for (int i = 1; i <= n; ++i) {
            int len = numberOfBits(i);
            ans = ((ans << len) % MOD + i) % MOD;
        }
        return ans;
    }
};
```

![Untitled](Number%20theory%2004047ba1cfe94ac89baae677375df8f2/Untitled%2018.png)

 

Problem 2

![Untitled](Number%20theory%2004047ba1cfe94ac89baae677375df8f2/Untitled%2019.png)

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N=2e5+10;

int hsh[N];
int multiples_cnt[N];
int main() {
	int n;
	cin>>n;
	for(int i=0;i<n;i++){
		int x;
		cin>>x;
		hsh[x]++;
	}

	for(int i=1;i<N;i++){
		for(int j=i;j<N;j+=i){
			multiples_cnt[i]+=hsh[j];
		}
	}
	int qq;
	cin>>qq;
	while(qq--){
		int p,q;
		cin>>p>>q;
		long long lcm=p*1LL*q/__gcd(p,q);
		long long ans=multiples_cnt[p]+multiples_cnt[q];
		if(lcm<N) ans-=multiples_cnt[lcm];
		cout<<ans<<"\n";
	}

}
```

> T primes are the number having exactly 3 divisor and they are squares of prime numbers
> 

## Compute the count of factors for numbers 1 to N in NlogN

```cpp
void fill_cnt_of_factors(int n){
    vi cnt(n+1);
    for(int i=1;i<=n;i++){
        for(int j=i;j<=n;j+=i){
            cnt[j]++;
        }
    }
    for(int i=1;i<=n;i++){
        cout<<"For i="<<i<<" cnt of factors: "<<cnt[i]<<endl;
    }

}
```

## Count number of pairs such that a[i]≠a[j] and a[i]%a[j]==0

```cpp
void solve(){
    int n;
    cin>>n;
    vi arr(n);
    inputArr(arr);
    unordered_map<int,int> cnt;
    for(auto it:arr) cnt[it]++;
    vi M(1e5+1);
    for(int i=1;i<=1e5;i++){
        for(int j=2*i;j<=1e5;j+=i){
            M[i]+=cnt[j];
        }
    }
    int ans=0;
    for(auto it:arr){
        ans+=M[it];
    }
    print(ans);
}

//input
7
2 2 3 6 7 42 42

//output
13
```