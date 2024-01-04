#pragma once
#include <string>
#include <vector>

class CSV
{
private:

public:
	CSV(void) {}
	~CSV(void) {}

	int Read(
		std::string file_name, 
		char delimiter, 
		std::vector<std::vector<std::string>>& rows);

	int Write(
		std::string file_name,
		char delimiter,
		std::vector<std::vector<std::string>>& rows);
};

