#include<iostream>
#include<string>
#include<fstream>
#include<time.h>
#include<sstream>
using namespace std;


/*
float Weierstrass() {

	float result_1 = 0;
	for (int j = 0; j <= 20; ++j) {
		result_1 += powf(0.5, j) * cosf(2 * 3.1415926 * powf(3, j) * 0.5);
	}
	return result_1;
}
float W_para = Weierstrass();
int main() {
	string str;
	printf("%f\n", 1.f);
	while (std::getline(cin, str)) {
		stringstream ss(str);
		float a;
		//float uns = 0;
		float uns = 418.9829 * 50;
		float temp_1 = INT_MAX, temp_2 = INT_MAX;
		int i = 1;
		float uns_1 = 0;
		while (ss >> a) {
			/*
			//case 1
			if (temp_1 == INT_MAX) {
			}
			else {
				uns += 100 * powf((temp_1 * temp_1 - a), 2) + powf(temp_1 - 1, 2);
			}
			temp_1 = a;


			//case 5:
		//uns = 1 + temp_1 / 4000 - temp_2;
		//for (int j = 0; j <= 20; ++j) {
		//	uns += powf(0.5, j) * cosf(2 * 3.1415926 * powf(3, j) * (a + 0.5));
		//}
		//uns = uns - 50 * W_para;

			//case 2:
			uns += a * a;
			uns_1 += cosf(2 * 3.1415926 * a);

			//case 0
			uns += a * a;
			++i;
			if (temp_1 == INT_MAX) {
				printf("can not begin now..%f\n", a);
			}
			else {
				uns += 100 * powf((powf(temp_1, 2) - a), 2) + powf(temp_1 - 1, 2);
			}
			temp_1 = a;
			* /
			uns -= a * sinf(sqrtf(fabsf(a)));
		}
		//uns = -20 * expf(-0.2 * sqrtf(uns / 50)) - expf(uns_1 / 50) + 20 + 2.718282;

		printf(
			"uns:%f\n", uns
		);

	}
}

int main() {
	int d;
	cin >> d;
	int indiv_perblock = (float)(-(40 + 3 * d + 2.f * (1.f / 3.f)) + sqrtf(powf(40 + 3.f * d + 2.f / 3, 2) - 8.f / 3 * (2 * d - 16 * 1024))) / (4.f / 3);
	int block_num = 511 / (2 * indiv_perblock) + 1;
	int stream_num = 82 / block_num;
	printf("inpl:%d, bn:%d, sn:%d\n", indiv_perblock, block_num, stream_num);
}
*/
/*
int main() {
	string str;
	float func[21], val[21];
	float mean_val[21];
	
	for (int i = 0; i < 21; ++i) {
		printf("%d\n", i);
		func[i] = 0;
		val[i] = 0;
		//cin >> mean_val[i];
	}
	//std::getline(cin, str);
	printf("0000000\n");
	int time = 0;
	while (std::getline(cin, str)) {
		if (str == "") {
			break;
		}
		time += 1;
		stringstream ss(str);
		float a;
		float uns = 0;
		float temp_1 = INT_MAX, temp_2 = INT_MAX;
		int i = 0;
		while (ss >> a) {
			func[i] += a;
			val[i] += (a - mean_val[i]) * (a - mean_val[i]);
			i++;
		}
		for (int j = 0; j < 21; ++j) {
			printf("%f,\n", func[j] / time);
		}
	}
	for (int j = 0; j < 21; ++j) {
		//printf("%f,\n", sqrtf(val[j] / time));
	}
}
*/
const int T = 5000, ITER = 1000 / 50 + 2, RUN = 20;
float record[RUN][T][ITER];
int main() {
	ifstream file;
	file.open("./ANO_Tnum/data_record_DE_T_NUM_5000.txt", ios::in);
	char* temp = new char[1000000];
	string str;
	int task_id = 0, run_id = 0;
	while (file.getline(temp , INT_MAX)) {
		str = temp;
		if (str.empty()) {
			printf("to be continued..%d, %d\n", task_id, run_id);
			task_id = 0;
			run_id += 1;
			continue;
		}
		stringstream ss(str);
		for (int i = 0; i < ITER; ++i) {

			ss >> record[run_id][task_id][i];
			//printf("%f, ", record[run_id][task_id][i]);
		}
		//printf("\n\n\n");
		task_id += 1;
	}
	float min[T], max[T];

	for (int i = 0; i < T; ++i) {
			min[i] = INT_MAX;
			max[i] = -1;
	}
	float result[ITER];
	for (int i = 0; i < ITER; ++i) {
		result[i] = 0;
	}
	fstream f;
	f.open("test.txt", ios::out);
	for (int k = 0; k < RUN; ++k) {
		for (int i = 0; i < T; ++i) {//任务
			for (int j = 0; j < ITER; ++j) {//迭代
				f << record[k][i][j] << ' ';
				//printf("%f ", record[k][i][j]);
			}
			f << endl;
			//printf("\n");
		}
		f << endl;
		//printf("\n");
	}
	f.close();
	for (int i = 0; i < T; ++i) {//任务
		for (int k = 0; k < RUN; ++k) {
			if (record[k][i][ITER - 2] < min[i]) {
				min[i] = record[k][i][ITER - 2];
			}
			if (record[k][i][ITER - 1] > max[i]) {
				max[i] = record[k][i][ITER - 1];
			}

		}
		for (int j = 0; j < ITER - 1; ++j) {//迭代
			//if (i < 100) {
				//printf("min:%f, max:%f, %d\n", min[i][j], max[i][j], j);
			//}
			for (int k = 0; k < RUN; ++k) {//循环数
				float temp = 0;
				if (max[i] - min[i] != 0) {
					temp = (record[k][i][j] - min[i]) / (max[i] - min[i]);
					
				}
				result[j] += temp;
			}
		}
	}
	f.open(string("./DE_DIMENSION/data_record_DE_T_NUM_5000_") + to_string(int(time(0))) + string(".txt"), ios::out);
	for (int i = 0; i < ITER - 1; ++i) {
		result[i] /= T * RUN;
		printf("%f, ", result[i]);
		//if (i <= 20) {
			f << result[i] << ", ";
		//}
	}
	f << endl;
	f.close();
	printf("endl");
	string a;
	cin >> a;
}