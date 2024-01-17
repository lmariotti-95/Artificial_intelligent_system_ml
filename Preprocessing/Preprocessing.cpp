#include <iostream>
#include <fstream>
#include <string>

#include "CSV.h"

using namespace std;

template <typename T>
vector<vector<T>> transpose_matrix(vector<vector<T>>& matrix)
{
  // Get the dimensions of the original matrix
  size_t rows = matrix.size();
  size_t cols = (rows > 0) ? matrix[0].size() : 0;

  // Create the transposed matrix
  vector<vector<T>> transposed(cols, vector<T>(rows));

  // Fill the transposed matrix
  for (size_t i = 0; i < rows; ++i) 
  {
    for (size_t j = 0; j < cols; ++j) 
    {
      transposed[j][i] = matrix[i][j];
    }
  }

  return transposed;
}

int main()
{
  string def_path = "C:\\Users\\Paola Diaz\\Desktop\\Lorenzo\\Artificial_intelligent_system_ml\\Assignment\\data_set";
  string in_file_name = def_path + "Company_Bankruptcy_Prediction.csv";
  string out_file_name = def_path + "processed.csv";

  vector<vector<string>> rows;
  CSV _csv = CSV();
  
  // Carico tutto il file
  _csv.Read(in_file_name, ',', rows);

  // Inizializzo la variabile di output
  //vector<vector<string>> output;

  auto attributes = transpose_matrix<string>(rows);

  for (int i = 1; i < attributes.size(); i++)
  {
    double max = stod(*max_element(attributes[i].begin(), attributes[i].end()));

    if (max > 1)
    {
      for (int j = 1; j < attributes[i].size(); j++)
      {
        double x = stod(attributes[i][j]);
        x /= max;
        attributes[i][j] = to_string(x);
      }
    }
  }

  auto output = transpose_matrix<string>(attributes);
  _csv.Write(out_file_name, ',', output);

  //for (int i = 0; i < rows[i].size(); i++)
  //{
  //  vector<double> v;
  //  double max = 1;
  //  for (int j = 1; j < rows.size(); j++)
  //  {
  //    double k = stod(rows[j][i]);
  //    v.push_back(k);
  //
  //    if (k > max)
  //      max = k;
  //  }
  //
  //  vector<string> new_row;
  //  for (double x : v)
  //  {
  //    if(max > 1)
  //      x /= max;
  //
  //    new_row.push_back(to_string(x));
  //  }
  //
  //  output.push_back(new_row);
  //}
  //
  //auto t_output = transpose_matrix<string>(output);
  //t_output.insert(t_output.begin(), rows[0]);
  //_csv.Write(out_file_name, ',', t_output);

  return 0;
}