/* Copyright 2021 Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ParticleCreationFunc.H"

#include "BinaryCollisionUtils.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/Algorithms/IsIn.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <string>

ParticleCreationFunc::ParticleCreationFunc (const std::string& collision_name,
                                            MultiParticleContainer const * const mypc):
    m_collision_type{BinaryCollisionUtils::get_collision_type(collision_name, mypc)}
{
    const amrex::ParmParse pp_collision_name(collision_name);

    if (m_collision_type == CollisionType::ProtonBoronToAlphasFusion)
    {
        // Proton-Boron fusion only produces alpha particles
        m_num_product_species = 1;
        // Proton-Boron fusion produces 3 alpha particles per fusion reaction
        m_num_products_host.push_back(3);
#ifndef AMREX_USE_GPU
        // On CPU, the device vector can be filled immediately
        m_num_products_device.push_back(3);
#endif
    }
    else if (m_collision_type == CollisionType::Recombination)
    {
        // Recombination only produces the recombined atom
        m_num_product_species = 1;
        m_num_products_host.push_back(1);
        //m_num_products_host.push_back(1);
#ifndef AMREX_USE_GPU
        // On CPU, the device vector can be filled immediately
        m_num_products_device.push_back(1);
        //m_num_products_device.push_back(1);
#endif
    }
    else if (m_collision_type == CollisionType::Dissociation)
    {
        // Recombination only produces the recombined atom
        m_num_product_species = 2; // could be more...
        m_num_products_host.push_back(2);
        m_num_products_host.push_back(1);
#ifndef AMREX_USE_GPU
        // On CPU, the device vector can be filled immediately
        m_num_products_device.push_back(2);
        m_num_products_device.push_back(1);
#endif
    }
    else if (m_collision_type == CollisionType::Ionization)
    {
        // x + y -> x + y+ + e but if x==e then only 2 products
        // otherwise is like: H + H -> H+ + e + H
        pp_collision_name.query("ionization_energy",m_energy_penalty);

        amrex::Vector<std::string> species;
        pp_collision_name.queryarr("species",species);

        amrex::Vector<std::string> product_species;
        pp_collision_name.queryarr("product_species",product_species);

        amrex::Vector<std::string> products;
        for (int i = 0; i< product_species.size(); i++){
            std::string out_species = product_species[i];
            
            if (utils::algorithms::is_in(species, out_species) && m_index_unionized_product<0){
                //we should track the 
                
                m_index_unionized_product = i;
            }
            auto it = std::find(products.begin(), products.end(), out_species);
            if (it==products.end()){
                // Then not found
                products.push_back(out_species);
                m_num_products_host.push_back(1);
                #ifndef AMREX_USE_GPU
                    // On CPU, the device vector can be filled immediately
                    m_num_products_device.push_back(1);

                #endif
            }
            else{
                size_t index = std::distance(products.begin(), it);
                m_num_products_host[index]++;
                #ifndef AMREX_USE_GPU
                    // On CPU, the device vector can be filled immediately
                    m_num_products_device[index]++;
                #endif
            }


        }
        m_num_product_species = (int)products.size();

    }
    else if ((m_collision_type == CollisionType::DeuteriumTritiumToNeutronHeliumFusion)
             || (m_collision_type == CollisionType::DeuteriumDeuteriumToProtonTritiumFusion)
             || (m_collision_type == CollisionType::DeuteriumDeuteriumToNeutronHeliumFusion))
    {
        m_num_product_species = 2;
        m_num_products_host.push_back(1);
        m_num_products_host.push_back(1);
#ifndef AMREX_USE_GPU
        // On CPU, the device vector can be filled immediately
        m_num_products_device.push_back(1);
        m_num_products_device.push_back(1);
#endif
    }
    else
    {
        WARPX_ABORT_WITH_MESSAGE("Unknown collision type in ParticleCreationFunc");
    }

#ifdef AMREX_USE_GPU
     m_num_products_device.resize(m_num_product_species);
     amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_num_products_host.begin(),
                           m_num_products_host.end(),
                           m_num_products_device.begin());
     amrex::Gpu::streamSynchronize();
#endif
}
