
#include<mutex>
#include<condition_variable>
#include<iostream>
using namespace std;

class semaphore{
	public:
	semaphore(int value = 1): count{value}, wakeups{0}{}

	void wait(){
		unique_lock<mutex> lock{mutex_s};
            //printf("count val:%d\n", count);
        if(--count < 0){
            condition.wait(lock, [&]()->bool{ return wakeups>0;});
            --wakeups;
        }
	}
    void signal(){
        lock_guard<mutex> lock{mutex_s};
        if(++count <= 0){
            ++wakeups;
            condition.notify_one();
        }
            //printf("count now:%d\n", count);
    }
    int getCount(){
        return count;
    }
	private:
	int count;
	int wakeups;
	mutex mutex_s;
	condition_variable condition;
};