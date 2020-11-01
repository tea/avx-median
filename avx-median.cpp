#include "avx-median.h"
#include <celero/Celero.h>
#include <random>
#include <cassert>
#include <iostream>
#include <iomanip>

static constexpr size_t data_size = 131069; // ~512 KB - fits in L2 cache; TODO would be nice to ensure there are no overreads
static constexpr size_t canary_size = 8;
static constexpr size_t output_data_size = data_size + 2 * canary_size;

float* input_data;
float* output_data;
float* raw_output_data;
float* golden_output_data;

void dump_reg(const char* const name, rf512 value)
{
	union
	{
		float f[16];
		r512f r;
	} values;
	store_to_address(values.f, value);
	std::cout << std::setw(10) << name << " = ";
	for (int i = 0; i < 15; ++i)
	{
		std::cout << std::setw(10) << values.f[i] << " | ";
	}
	std::cout << values.f[15] << "\n";
}

void dump_reg(const char* const name, ri512 value)
{
	union
	{
		int32_t f[16];
		ri512 r;
	} values;
	store_to_address(values.f, value);
	std::cout << std::setw(10) << name << " = ";
	for (int i = 0; i < 15; ++i)
	{
		std::cout << std::setw(10) << values.f[i] << " | ";
	}
	std::cout << values.f[15] << "\n";
}

static void validate(void(*method)(const float*, float*, size_t))
{
	std::fill_n((uint8_t*)raw_output_data, output_data_size * sizeof(raw_output_data[0]), 0xCD);
	method(input_data, output_data, data_size);
	if (!std::equal(raw_output_data, raw_output_data + output_data_size, golden_output_data))
	{
		assert(false);
		std::cerr << "Validation failed\n";
		for (int i = 0; i < canary_size; ++i)
			std::cerr << "#" << i << ": " << raw_output_data[i] << "\t" << golden_output_data[i] << "\n";
		for (int i = canary_size; i < canary_size + 32; ++i)
			std::cerr << "#" << i << ": " << raw_output_data[i] << "\t" << golden_output_data[i] << "\t" << input_data[i - canary_size] << "\n";
		for (int i = output_data_size - 16; i < output_data_size; ++i)
			std::cerr << "#" << i << ": " << raw_output_data[i] << "\t" << golden_output_data[i] << "\n";
		exit(1);
	}
}

static void init()
{
	std::mt19937 RandomDevice;
	std::uniform_real_distribution<float> Distribution{ -1, 1 };
	auto alloc = [](size_t size) -> float* {return reinterpret_cast<float*>(::operator new[](size * sizeof(float), std::align_val_t{ 16 })); };
	input_data = alloc(data_size);
	raw_output_data = alloc(output_data_size);
	golden_output_data = alloc(output_data_size);

	std::generate_n(input_data, data_size, [&]() {return Distribution(RandomDevice); });

	std::fill_n((uint8_t*)raw_output_data, output_data_size * sizeof(raw_output_data[0]), 0xCD);
	output_data = raw_output_data + canary_size;
	median_Cpp(input_data, output_data, data_size);
	std::copy_n(raw_output_data, output_data_size, golden_output_data);
}

static void validate()
{
	validate(median_Step0);
	validate(median_Step1);
	validate(median_Step2);
	validate(median_Step3);
	validate(median_Parallel);
	validate(median_Parallel_avx2);
	validate(median_Parallel_step1);
}

int main(int argc, char** argv)
{
	init();
	validate();
	celero::Run(argc, argv);
	return 0;
}

static constexpr auto BENCH_SAMPLES = 30;
static constexpr auto BENCH_ITERATIONS = 1000;

#if 0
BASELINE(Median, Cpp, BENCH_SAMPLES, BENCH_ITERATIONS)
{
	median_Cpp(input_data, output_data, data_size);
}

BENCHMARK(Median, Step0, BENCH_SAMPLES, BENCH_ITERATIONS)
{
	median_Step0(input_data, output_data, data_size);
}
#else
BASELINE(Median, Step0, BENCH_SAMPLES, BENCH_ITERATIONS)
{
	median_Step0(input_data, output_data, data_size);
}
#endif

BENCHMARK(Median, Step1, BENCH_SAMPLES, BENCH_ITERATIONS)
{
	median_Step1(input_data, output_data, data_size);
}

BENCHMARK(Median, Step2, BENCH_SAMPLES, BENCH_ITERATIONS)
{
	median_Step2(input_data, output_data, data_size);
}

BENCHMARK(Median, Step3, BENCH_SAMPLES, BENCH_ITERATIONS)
{
	median_Step3(input_data, output_data, data_size);
}

BENCHMARK(Median, Parallel, BENCH_SAMPLES, BENCH_ITERATIONS)
{
	median_Parallel(input_data, output_data, data_size);
}

#if 0
BENCHMARK(Median, ParallelAVX2, BENCH_SAMPLES, BENCH_ITERATIONS)
{
	median_Parallel_avx2(input_data, output_data, data_size);
}
#endif

BENCHMARK(Median, ParallelStep1, BENCH_SAMPLES, BENCH_ITERATIONS)
{
	median_Parallel_step1(input_data, output_data, data_size);
}

BENCHMARK(Median, Memcpy, BENCH_SAMPLES, BENCH_ITERATIONS)
{
	float* psrc = input_data;
	float* pdst = output_data;
	size_t size = data_size;
	while (size >= 16)
	{
		rf512 curr = load_value(psrc[0]);
		store_to_address(pdst, curr);
		psrc += 16;
		pdst += 16;
		size -= 16;
	}
}
