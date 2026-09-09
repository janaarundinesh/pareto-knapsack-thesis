#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <mpi.h>

#include "brute_force_maxima.hpp"
#include "dominance.hpp"


// ============================================================
// Convert Item -> flat vector
// ============================================================

std::vector<double> FlattenItems(
    const std::vector<Item>& items,
    size_t dimensions)
{
    std::vector<double> data(
        items.size() * dimensions
    );

    for (size_t i = 0; i < items.size(); ++i)
    {
        for (size_t j = 0; j < dimensions; ++j)
        {
            data[i * dimensions + j] =
                items[i].coords[j];
        }
    }

    return data;
}


// ============================================================
// Convert flat vector -> Item
// ============================================================

std::vector<Item> UnflattenItems(
    const std::vector<double>& data,
    size_t number_of_items,
    size_t dimensions)
{
    std::vector<Item> items(number_of_items);

    for (size_t i = 0; i < number_of_items; ++i)
    {
        items[i].coords.resize(dimensions);

        for (size_t j = 0; j < dimensions; ++j)
        {
            items[i].coords[j] =
                data[i * dimensions + j];
        }
    }

    return items;
}

std::vector<Item> MergeMaxima(
    const std::vector<Item>& existing,
    const std::vector<Item>& incoming,
    size_t d)
{
    std::vector<Item> result;
    result.reserve(existing.size() + incoming.size());

    // Keep existing points that are not dominated by any incoming point
    for (const auto& e : existing)
    {
        bool dominated = false;

        for (const auto& n : incoming)
        {
            if (Dominates(n, e, d))
            {
                dominated = true;
                break;
            }
        }

        if (!dominated)
        {
            result.push_back(e);
        }
    }

    // Keep incoming points that are not dominated by any (original) existing point
    for (const auto& n : incoming)
    {
        bool dominated = false;

        for (const auto& e : existing)
        {
            if (Dominates(e, n, d))
            {
                dominated = true;
                break;
            }
        }

        if (!dominated)
        {
            result.push_back(n);
        }
    }

    return result;
}


int main(int argc, char** argv)
{
    int rank;
    int size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // ========================================================
    // 1. Rank 0 reads the dataset
    // ========================================================

    uint64_t dimensions = 0;
    uint64_t num_points = 0;

    uint64_t info[2] = {0, 0};

    std::vector<double> all_data;

    if (rank == 0)
    {
        std::string filename =
            "../DataSets/10D_Data/Test5.txt";

        std::ifstream file(filename);

        if (!file)
        {
            std::cerr
                << "Could not open dataset: "
                << filename
                << std::endl;

            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        file >> dimensions >> num_points;

        // ====================================================
        // 2. Read all coords
        // ====================================================

        all_data.resize(num_points * dimensions);

        for (uint64_t i = 0; i < num_points; ++i)
        {
            for (uint64_t j = 0; j < dimensions; ++j)
            {
                file >> all_data[
                    i * dimensions + j
                ];
            }
        }

        file.close();

        info[0] = dimensions;
        info[1] = num_points;
    }

    MPI_Bcast(info, 2, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    dimensions = info[0];
    num_points = info[1];


    // ========================================================
    // 3. Calculate how many points each process gets
    // ========================================================

    uint64_t base_points = num_points / size;

    uint64_t remainder = num_points % size;

    uint64_t local_points = base_points;

    if (static_cast<uint64_t>(rank) < remainder)
    {
        local_points++;
    }


    // ========================================================
    // 4. Prepare send counts and displacements
    // ========================================================

    std::vector<int> send_counts(size); // Number of values from the data to send to each process
    std::vector<int> displacements(size); // Starting index in the all_data array for each process

    uint64_t current_offset = 0;

    for (int p = 0; p < size; ++p)
    {
        uint64_t points_for_process =
            base_points;

        if (static_cast<uint64_t>(p) < remainder)
        {
            points_for_process++;
        }

        send_counts[p] = static_cast<int>(points_for_process * dimensions);

        displacements[p] = static_cast<int>(current_offset * dimensions);

        current_offset += points_for_process;
    }


    // ========================================================
    // 5. Allocate local data
    // ========================================================

    std::vector<double> local_data(local_points * dimensions);


    // ========================================================
    // 6. FAN-OUT
    //
    // Rank 0 distributes the data to all processes.
    //
    // ========================================================

    MPI_Scatterv(
        all_data.data(),
        send_counts.data(),
        displacements.data(),
        MPI_DOUBLE,

        local_data.data(),
        static_cast<int>(local_data.size()),
        MPI_DOUBLE,

        0,
        MPI_COMM_WORLD
    );


    // ========================================================
    // 7. Convert local flat data into Item objects
    // ========================================================

    std::vector<Item> local_items =
        UnflattenItems(
            local_data,
            local_points,
            dimensions
        );


    // ========================================================
    // 8. Calculate LOCAL MAXIMA
    // ========================================================

    std::vector<Item> local_maxima = BruteForceMaxima(local_items,dimensions);


    // ========================================================
    // 9. FAN-IN
    // ========================================================

    int step = 1;

    while (step < size)
    {
        // ----------------------------------------------------
        // RECEIVERS
        // ----------------------------------------------------

        if (rank % (2 * step) == 0)
        {
            int sender = rank + step;

            if (sender < size)
            {
                // =================================================
                // Receive number of maxima
                // =================================================

                int received_count = 0;

                MPI_Recv(
                    &received_count,
                    1,
                    MPI_INT,
                    sender,
                    0,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE
                );


                // =================================================
                // Receive maxima data
                // =================================================

                std::vector<double> received_data(
                    static_cast<size_t>(
                        received_count
                    ) * dimensions
                );

                MPI_Recv(
                    received_data.data(),
                    static_cast<int>(
                        received_data.size()
                    ),
                    MPI_DOUBLE,
                    sender,
                    1,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE
                );

                std::vector<Item> received_maxima = UnflattenItems(received_data, received_count, dimensions);

                // =================================================
                // Both sides are already individually non-dominated,
                // so only cross-compare existing vs newly received
                // rather than re-running brute force on everything.
                // =================================================

                local_maxima = MergeMaxima(local_maxima, received_maxima, dimensions);
            }
        }

        // ----------------------------------------------------
        // SENDERS
        // ----------------------------------------------------

        else
        {
            int receiver = rank - step;

            int count =
                static_cast<int>(
                    local_maxima.size()
                );

            std::vector<double> maxima_data = FlattenItems(local_maxima, dimensions);

            MPI_Request requests[2];

            MPI_Isend(
                &count,
                1,
                MPI_INT,
                receiver,
                0,
                MPI_COMM_WORLD,
                &requests[0]
            );

            MPI_Isend(
                maxima_data.data(),
                static_cast<int>(
                    maxima_data.size()
                ),
                MPI_DOUBLE,
                receiver,
                1,
                MPI_COMM_WORLD,
                &requests[1]
            );

            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

            break;
        }

        step *= 2;
    }

    if (rank == 0)
    {
        std::cout << "\nNumber of processes: " << size << std::endl;

        std::cout << "\nInitial number of points: " << num_points << std::endl;

        std::cout << "\nFinal maxima: " << local_maxima.size() << std::endl;
    }

    MPI_Finalize();

    return 0;
}