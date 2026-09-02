#include<iostream>
#include "brute_force_maxima.hpp"
#include "dominance.hpp"
#include<mpi.h>

std::vector<Item> BruteForceMaxima(
    const std::vector<Item>& points,
    size_t d)
{
    std::vector<Item> result;

    for (size_t i = 0; i < points.size(); ++i)
    {
        bool dominated = false;

        for (size_t j = 0; j < points.size(); ++j)
        {
            if (i == j) continue;

            if (Dominates(points[j], points[i], d))
            {
                dominated = true;
                break;
            }
        }

        if (!dominated)
        {
            result.push_back(points[i]);
        }
    }

    return result;
}

int main (int argc, char** argv){

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        

        std::string filename = "../DataSets/5D_Data/Test1.txt";

        std::ifstream file(filename);

        if (!file)
        {
            std::cerr << "Could not open dataset: "
                    << filename << '\n';

            return {};
        }

        size_t dimensions;
        size_t num_points;

        file >> dimensions >> num_points;

        std::vector<Item> items(num_points);



        for (size_t i = 0; i < num_points; ++i)
        {
            items[i].coords.resize(dimensions);

            for (size_t d = 0; d < dimensions; ++d)
            {
                file >> items[i].coords[d];
            }
        }
        file.close();
    }

    size_t local_data_size = num_points / size; // Calculate the size of data each process will handle

    // Determine the start and end indices for each process

    if (num_points % size == 0) {
        size_t start_index = rank * local_data_size;
        size_t end_index = (start_index + local_data_size) - 1;
    } else {
        size_t start_index = rank * local_data_size;
        size_t end_index = (start_index + local_data_size);
        if (rank == size - 1) {
            end_index = num_points - 1;
        }
    }
    
    

    MPI_Finalize();
    return 0;
}