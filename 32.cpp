#include<iostream>

using namespace std;

// class Solution {
// public:
//     int* longestValidParentheses(string s) {
//         int length = sizeof(s);
//         int dp[length]={0};
//         for (int i=1; i<length; i++)
//         {
//             if (s[i] == '(')
//             {
//                 dp[i] = 0;
//             }
//             else
//             {
//                 if (s[i-1] == '(')
//                 {
//                     dp[i] = i>2?2+dp[i-2]:2;
//                 }
//                 else{
//                     if (i-dp[i-1]-1>=0)
//                     {
//                         dp[i] = i-dp[i-1]-1>0?dp[i-1]+2+dp[i-dp[i-1]-2]:dp[i-1]+2;
//                     }
//                     else{dp[i] = 0;}
//                 }
//             }
//         }
//         return dp;
//     }
// };
class Solution {
public:
    int longestValidParentheses(string s) {
        stack<int> stk;
        stk.push(-1);
        int maxL=0;
        for(int i=0;i<s.size();i++)
        {
            int t=stk.top();
            if(t!=-1&&s[i]==')'&&s[t]=='(')
            {
                stk.pop();
                maxL=max(maxL,i-stk.top());
            }
            else
                stk.push(i);
        }
        return maxL;
    }
};
