#include "CSV.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

std::vector<std::string> split_line(const std::string& line, char delimiter) 
{
  std::vector<std::string> tokens;
  std::istringstream tokenStream(line);
  std::string token;

  while (std::getline(tokenStream, token, delimiter)) 
  {
    tokens.push_back(token);
  }

  return tokens;
}

int CSV::Read(
  std::string file_name, 
  char delimiter, 
  std::vector<std::vector<std::string>> &rows)
{
  std::ifstream file(file_name);

  if (!file.is_open()) 
  {
    std::cerr << "Unable to open the file.\n";
    return 1;
  }

  std::string line;
  while (std::getline(file, line)) 
  {
    rows.push_back(split_line(line, delimiter));
  }

  file.close();

	return 0;
}

int CSV::Write(
  std::string file_name,
  char delimiter,
  std::vector<std::vector<std::string>>& rows)
{
  std::ofstream file(file_name);

  if (!file.is_open())
  {
    std::cerr << "Unable to open the file.\n";
    return 1;
  }

  for (int i = 0; i < rows.size(); i++)
  {
    for (int j = 0; j < rows[i].size(); j++)
    {
      file << rows[i][j];
      if (j != rows[i].size() - 1)
      {
        file << delimiter;
      }
    }

    file << std::endl;
  }

  file.close();
}