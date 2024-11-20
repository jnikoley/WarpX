#ifndef WARPX_DISSOCIATION_COLLISION_FUNC_H_
#define WARPX_DISSOCIATION_COLLISION_FUNC_H_

#include "Particles/Collision/CollisionBase.H"
#include "Utils/Parser/ParserUtils.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/ParticleUtils.H"
#include <AMReX.H>
#include <AMReX_REAL.H>




namespace {
    // Define shortcuts for frequently-used type names
    using SoaData_type = typename WarpXParticleContainer::ParticleTileType::ParticleTileDataType;
    using ParticleType = typename WarpXParticleContainer::ParticleType;
    using ParticleTileType = typename WarpXParticleContainer::ParticleTileType;
    using ParticleTileDataType = typename ParticleTileType::ParticleTileDataType;
    using ParticleBins = amrex::DenseBins<ParticleTileDataType>;
    using index_type = typename ParticleBins::index_type;

    /**
     * \brief This function initializes the momentum of the particles created by a dsmc 
     * interaction between any incoming particle and a molecule. The incident particle has its 
     * energy reduced by the penalty and the other particles have the same momentum as the 
     * original particle
     * @param[in] soa_1 struct of array data of the first colliding species (can be either proton
     * or boron)
     * @param[in] soa_2 struct of array data of the second colliding species (this is the molecule which
     * will be split)
     * @param[out] soa_product struct of array data of the alpha species
     * @param[in] idx_1 index of first colliding macroparticle
     * @param[in] idx_2 index of second colliding macroparticle
     * @param[in] idx_product_start index of first produced alpha macroparticle
     * @param[in] m1 mass of first colliding species
     * @param[in] m2 mass of second colliding species
     * @param[in] engine the random engine
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void DissociationInitializeMomentum2Prod (
                            const SoaData_type& soa_1, const SoaData_type& soa_2,
                            const SoaData_type* soa_products,
                            const index_type& idx_1, const index_type& idx_2,
                            const index_type& idx_prod_0,const index_type& idx_prod_1,
                            const index_type& idx_unchanged_product,
                            const amrex::ParticleReal& m1, const amrex::ParticleReal& m2,
                            const amrex::ParticleReal* product_mass,
                            amrex::ParticleReal energy_cost,
                            const amrex::RandomEngine& engine)
    {


        using std::sqrt;
        using namespace amrex::literals;
    
        // precalculate often used value
        constexpr auto c2 = PhysConst::c * PhysConst::c;

        index_type idx_prod_unchanged = idx_prod_0;
        index_type idx_prod_a = idx_prod_1;
        index_type idx_a = 1;

        if (idx_unchanged_product==1){
            idx_prod_unchanged = idx_prod_1;
            idx_prod_a = idx_prod_0;
            idx_a = 0;
        } 
        
        auto& ux1 = soa_1.m_rdata[PIdx::ux][idx_1];
        auto& uy1 = soa_1.m_rdata[PIdx::uy][idx_1];
        auto& uz1 = soa_1.m_rdata[PIdx::uz][idx_1];
        auto& ux2 = soa_2.m_rdata[PIdx::ux][idx_2];
        auto& uy2 = soa_2.m_rdata[PIdx::uy][idx_2];
        auto& uz2 = soa_2.m_rdata[PIdx::uz][idx_2];
        
        
        auto& uxprod0 = soa_products[idx_unchanged_product].m_rdata[PIdx::ux][idx_prod_unchanged];
        auto& uyprod0 = soa_products[idx_unchanged_product].m_rdata[PIdx::uy][idx_prod_unchanged];
        auto& uzprod0 = soa_products[idx_unchanged_product].m_rdata[PIdx::uz][idx_prod_unchanged];
        auto& uxprod1 = soa_products[idx_unchanged_product].m_rdata[PIdx::ux][idx_prod_unchanged + 1];
        auto& uyprod1 = soa_products[idx_unchanged_product].m_rdata[PIdx::uy][idx_prod_unchanged + 1];
        auto& uzprod1 = soa_products[idx_unchanged_product].m_rdata[PIdx::uz][idx_prod_unchanged + 1];


        amrex::ParticleReal u2;
        if (std::abs(m1-(2.0_prt*product_mass[idx_a]))<1e-28){
            // Then source particle 1 is the source particle
        
            // Fill momentum of product species (note that we actually
            // create 6 products, 2 at the position of each incident particle)
            soa_products[idx_a].m_rdata[PIdx::ux][idx_prod_a]   = ux1;
            soa_products[idx_a].m_rdata[PIdx::uy][idx_prod_a]   = uy1;
            soa_products[idx_a].m_rdata[PIdx::uz][idx_prod_a]   = uz1;
            soa_products[idx_a].m_rdata[PIdx::ux][idx_prod_a+1] = ux1;
            soa_products[idx_a].m_rdata[PIdx::uy][idx_prod_a+1] = uy1;
            soa_products[idx_a].m_rdata[PIdx::uz][idx_prod_a+1] = uz1;
            soa_products[idx_a].m_rdata[PIdx::ux][idx_prod_a+2] = ux1;
            soa_products[idx_a].m_rdata[PIdx::uy][idx_prod_a+2] = uy1;
            soa_products[idx_a].m_rdata[PIdx::uz][idx_prod_a+2] = uz1;
            soa_products[idx_a].m_rdata[PIdx::ux][idx_prod_a+3] = ux1;
            soa_products[idx_a].m_rdata[PIdx::uy][idx_prod_a+3] = uy1;
            soa_products[idx_a].m_rdata[PIdx::uz][idx_prod_a+3] = uz1;


            u2 = (ux2 * ux2) + (uy2 * uy2) + (uz2 * uz2);
        } else if(std::abs(m2-(2.0_prt*product_mass[idx_a]))<1e-28) {
            // Then source particle 2 is the source particle


            // Fill momentum of product species (note that we actually
            // create 6 products, 2 at the position of each incident particle)
            soa_products[idx_a].m_rdata[PIdx::ux][idx_prod_a]   = ux2;
            soa_products[idx_a].m_rdata[PIdx::uy][idx_prod_a]   = uy2;
            soa_products[idx_a].m_rdata[PIdx::uz][idx_prod_a]   = uz2;
            soa_products[idx_a].m_rdata[PIdx::ux][idx_prod_a+1] = ux2;
            soa_products[idx_a].m_rdata[PIdx::uy][idx_prod_a+1] = uy2;
            soa_products[idx_a].m_rdata[PIdx::uz][idx_prod_a+1] = uz2;
            soa_products[idx_a].m_rdata[PIdx::ux][idx_prod_a+2] = ux2;
            soa_products[idx_a].m_rdata[PIdx::uy][idx_prod_a+2] = uy2;
            soa_products[idx_a].m_rdata[PIdx::uz][idx_prod_a+2] = uz2;
            soa_products[idx_a].m_rdata[PIdx::ux][idx_prod_a+3] = ux2;
            soa_products[idx_a].m_rdata[PIdx::uy][idx_prod_a+3] = uy2;
            soa_products[idx_a].m_rdata[PIdx::uz][idx_prod_a+3] = uz2;

            u2 = (ux1 * ux1) + (uy1 * uy1) + (uz1 * uz1);
        } else{
            amrex::Abort("2 Particle Dissociation does not conserve mass");
        }
        amrex::ParticleReal Eout;
        ParticleUtils::getEnergy(u2, m1, Eout);

        Eout = ((Eout * PhysConst::q_e) - energy_cost)  / product_mass[idx_a];
        
        const amrex::ParticleReal uprod0 =  sqrt((2.0_prt* c2 * Eout) + (Eout*Eout))/PhysConst::c;

        // randomize the direction of both of the dissociation causing particles
        // There are two because this uses DSMC
        ParticleUtils::RandomizeVelocity(uxprod0,uyprod0,uzprod0,uprod0,engine);
        ParticleUtils::RandomizeVelocity(uxprod1,uyprod1,uzprod1,uprod0,engine);
    }

    /**
     * \brief This function initializes the momentum of the particles created by a dsmc 
     * interaction between any incoming particle and a molecule. The incident particle has its 
     * energy reduced by the penalty and the other particles have the same momentum as the 
     * original particle
     * @param[in] soa_1 struct of array data of the first colliding species (can be either proton
     * or boron)
     * @param[in] soa_2 struct of array data of the second colliding species (this is the molecule which
     * will be split)
     * @param[out] soa_product struct of array data of the alpha species
     * @param[in] idx_1 index of first colliding macroparticle
     * @param[in] idx_2 index of second colliding macroparticle
     * @param[in] idx_product_start index of first produced alpha macroparticle
     * @param[in] m1 mass of first colliding species
     * @param[in] m2 mass of second colliding species
     * @param[in] engine the random engine
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void DissociationInitializeMomentum3Prod (
                            const SoaData_type& soa_1, const SoaData_type& soa_2,
                            const SoaData_type* soa_products,
                            const index_type& idx_1, const index_type& idx_2,
                            const index_type& idx_prod_0,const index_type& idx_prod_1,
                            const index_type& idx_prod_2,
                            const index_type& idx_unchanged_product,
                            const amrex::ParticleReal& m1, const amrex::ParticleReal& m2,
                            const amrex::ParticleReal* product_mass,
                            amrex::ParticleReal energy_cost,
                            const amrex::RandomEngine& engine)
    {


        using std::sqrt;
        using namespace amrex::literals;
    
        // precalculate often used value
        constexpr auto c2 = PhysConst::c * PhysConst::c;

        index_type idx_prod_unchanged = idx_prod_0;
        index_type idx_prod_a = idx_prod_1;
        index_type idx_prod_b = idx_prod_2;
        index_type idx_a = 1;
        index_type idx_b = 2;
        if (idx_unchanged_product==1){
            idx_prod_unchanged = idx_prod_1;
            idx_prod_a = idx_prod_0;
            idx_prod_b = idx_prod_2;
            idx_a = 0;
            idx_b = 2;
        } else if (idx_unchanged_product==2){
            idx_prod_unchanged = idx_prod_2;
            idx_prod_a = idx_prod_0;
            idx_prod_b = idx_prod_1;
            idx_a = 0;
            idx_b = 1;
        }
        
        auto& ux1 = soa_1.m_rdata[PIdx::ux][idx_1];
        auto& uy1 = soa_1.m_rdata[PIdx::uy][idx_1];
        auto& uz1 = soa_1.m_rdata[PIdx::uz][idx_1];
        auto& ux2 = soa_2.m_rdata[PIdx::ux][idx_2];
        auto& uy2 = soa_2.m_rdata[PIdx::uy][idx_2];
        auto& uz2 = soa_2.m_rdata[PIdx::uz][idx_2];
        
        
        auto& uxprod0 = soa_products[idx_unchanged_product].m_rdata[PIdx::ux][idx_prod_unchanged];
        auto& uyprod0 = soa_products[idx_unchanged_product].m_rdata[PIdx::uy][idx_prod_unchanged];
        auto& uzprod0 = soa_products[idx_unchanged_product].m_rdata[PIdx::uz][idx_prod_unchanged];
        auto& uxprod1 = soa_products[idx_unchanged_product].m_rdata[PIdx::ux][idx_prod_unchanged + 1];
        auto& uyprod1 = soa_products[idx_unchanged_product].m_rdata[PIdx::uy][idx_prod_unchanged + 1];
        auto& uzprod1 = soa_products[idx_unchanged_product].m_rdata[PIdx::uz][idx_prod_unchanged + 1];


        amrex::ParticleReal u2;
        if (std::abs(m1-(product_mass[idx_a] + product_mass[idx_b]))<1e-28){
            // Then source particle 1 is the source particle
        
            // Fill momentum of product species (note that we actually
            // create 6 products, 2 at the position of each incident particle)
            soa_products[idx_a].m_rdata[PIdx::ux][idx_prod_a]   = ux1;
            soa_products[idx_a].m_rdata[PIdx::uy][idx_prod_a]   = uy1;
            soa_products[idx_a].m_rdata[PIdx::uz][idx_prod_a]   = uz1;
            soa_products[idx_a].m_rdata[PIdx::ux][idx_prod_a+1] = ux1;
            soa_products[idx_a].m_rdata[PIdx::uy][idx_prod_a+1] = uy1;
            soa_products[idx_a].m_rdata[PIdx::uz][idx_prod_a+1] = uz1;

            soa_products[idx_b].m_rdata[PIdx::ux][idx_prod_b] = ux1;
            soa_products[idx_b].m_rdata[PIdx::uy][idx_prod_b] = uy1;
            soa_products[idx_b].m_rdata[PIdx::uz][idx_prod_b] = uz1;
            soa_products[idx_b].m_rdata[PIdx::ux][idx_prod_b+1] = ux1;
            soa_products[idx_b].m_rdata[PIdx::uy][idx_prod_b+1] = uy1;
            soa_products[idx_b].m_rdata[PIdx::uz][idx_prod_b+1] = uz1;


            u2 = (ux2 * ux2) + (uy2 * uy2) + (uz2 * uz2);
        } else if (std::abs(m2-(product_mass[idx_a] + product_mass[idx_b]))<1e-28){
            // Then source particle 2 is the source particle

            
            // Fill momentum of product species (note that we actually
            // create 6 products, 2 at the position of each incident particle)
            soa_products[idx_a].m_rdata[PIdx::ux][idx_prod_a]   = ux2;
            soa_products[idx_a].m_rdata[PIdx::uy][idx_prod_a]   = uy2;
            soa_products[idx_a].m_rdata[PIdx::uz][idx_prod_a]   = uz2;
            soa_products[idx_a].m_rdata[PIdx::ux][idx_prod_a+1] = ux2;
            soa_products[idx_a].m_rdata[PIdx::uy][idx_prod_a+1] = uy2;
            soa_products[idx_a].m_rdata[PIdx::uz][idx_prod_a+1] = uz2;

            soa_products[idx_b].m_rdata[PIdx::ux][idx_prod_b] = ux2;
            soa_products[idx_b].m_rdata[PIdx::uy][idx_prod_b] = uy2;
            soa_products[idx_b].m_rdata[PIdx::uz][idx_prod_b] = uz2;
            soa_products[idx_b].m_rdata[PIdx::ux][idx_prod_b+1] = ux2;
            soa_products[idx_b].m_rdata[PIdx::uy][idx_prod_b+1] = uy2;
            soa_products[idx_b].m_rdata[PIdx::uz][idx_prod_b+1] = uz2;

            u2 = (ux1 * ux1) + (uy1 * uy1) + (uz1 * uz1);
        } else{
            amrex::Abort("3 Particle Dissociation does not conserve mass");
        }
        amrex::ParticleReal Eout;
        ParticleUtils::getEnergy(u2, m1, Eout);

        Eout = ((Eout * PhysConst::q_e) - energy_cost)  / product_mass[idx_a];
        
        const amrex::ParticleReal uprod0 =  sqrt((2.0_prt* c2 * Eout) + (Eout*Eout))/PhysConst::c;

        // randomize the direction of both of the dissociation causing particles
        // There are two because this uses DSMC
        ParticleUtils::RandomizeVelocity(uxprod0,uyprod0,uzprod0,uprod0,engine);
        ParticleUtils::RandomizeVelocity(uxprod1,uyprod1,uzprod1,uprod0,engine);
    }

}


#endif // WARPX_DISSOCIATION_COLLISION_FUNC_H_