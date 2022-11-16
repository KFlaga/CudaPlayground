#include "nanobench.h"
#include "bench_helpers.h"
#include <regex>

struct BenchmarkEntry
{
	std::string name;
	std::function<void()> fun;
};

struct BenchRegistry
{
	static BenchRegistry& instance()
	{
		static BenchRegistry r{};
		return r;
	}

	std::vector<BenchmarkEntry> benchmarks;

	void run(std::string filter = "")
	{
		for (auto& b : benchmarks)
		{
			if (filter == "")
			{
				b.fun();
			}
			else
			{
				std::regex r(filter, std::regex_constants::ECMAScript);
				if (std::regex_search(b.name, r))
				{
					b.fun();
				}
			}
		}
	}
};

void registerBench(std::string name, std::function<void()> fun)
{
	BenchRegistry::instance().benchmarks.push_back({ name, fun });
}


int main()
{
	BenchRegistry::instance().run();
}
