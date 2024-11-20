#ifndef WARPX_RECOMBINATION_COLLISION_FUNC_H_
#define WARPX_RECOMBINATION_COLLISION_FUNC_H_

#include "Particles/Collision/CollisionBase.H"
#include "Utils/Parser/ParserUtils.H"
#include "Particles/WarpXParticleContainer.H"
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
     * \brief This function initializes the momentum of the alpha particles produced from
     * proton-boron fusion. The momentum is initialized by assuming that the fusion of a proton
     * with a boron nucleus into 3 alphas takes place in two steps. In the first step, the proton
     * and the boron fuse into an excited beryllium nucleus and an alpha particle. In the second step, the
     * excited beryllium decays into two alpha particles. The first step produces ~5.56 MeV of kinetic
     * energy while the second step produces ~3.12 MeV of kinetic energy. This two-step process
     * is considered to be the dominant process of proton+boron fusion into alphas (see
     * D.R. Tilley et al. / Nuclear Physics A 745 (2004) 155–362).
     * For each step, we assume in this function that the particles are emitted isotropically in
     * the corresponding center of mass frame (center of mass frame of proton + boron for the
     * creation of first alpha+beryllium and rest frame of beryllium for the creation of second and
     * third alphas). This isotropic assumption is exact for the second step but is only an
     * approximation for the first step.
     *
     * @param[in] soa_1 struct of array data of the first colliding species (can be either proton
     * or boron)
     * @param[in] soa_2 struct of array data of the second colliding species (can be either proton
     * or boron)
     * @param[out] soa_product struct of array data of the alpha species
     * @param[in] idx_1 index of first colliding macroparticle
     * @param[in] idx_2 index of second colliding macroparticle
     * @param[in] idx_product_start index of first produced alpha macroparticle
     * @param[in] m1 mass of first colliding species
     * @param[in] m2 mass of second colliding species
     * @param[in] engine the random engine
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void RecombinationInitializeMomentum (
                            const SoaData_type& soa_1, const SoaData_type& soa_2,
                            SoaData_type& soa_product,
                            const index_type& idx_1, const index_type& idx_2,
                            const index_type& idx_product_start,
                            const amrex::ParticleReal& m1, const amrex::ParticleReal& m2,
                            const amrex::RandomEngine& engine)
    {


        using namespace amrex::literals;
    

        const amrex::ParticleReal px1 = m1 * soa_1.m_rdata[PIdx::ux][idx_1];
        const amrex::ParticleReal py1 = m1 * soa_1.m_rdata[PIdx::uy][idx_1];
        const amrex::ParticleReal pz1 = m1 * soa_1.m_rdata[PIdx::uz][idx_1];

        const amrex::ParticleReal px2 = m2 * soa_2.m_rdata[PIdx::ux][idx_2];
        const amrex::ParticleReal py2 = m2 * soa_2.m_rdata[PIdx::uy][idx_2];
        const amrex::ParticleReal pz2 = m2 * soa_2.m_rdata[PIdx::uz][idx_2];


        soa_product.m_rdata[PIdx::ux][idx_product_start] = (px1 + px2)/(m1 + m2);
        soa_product.m_rdata[PIdx::uy][idx_product_start] = (py1 + py2)/(m1 + m2);
        soa_product.m_rdata[PIdx::uz][idx_product_start] = (pz1 + pz2)/(m1 + m2);

        //Fill momentum of product species (note that we actually
        // create 2 products, 2 at the position of each incident particle)
        soa_product.m_rdata[PIdx::ux][idx_product_start + 1] = soa_product.m_rdata[PIdx::ux][idx_product_start];
        soa_product.m_rdata[PIdx::uy][idx_product_start + 1] = soa_product.m_rdata[PIdx::uy][idx_product_start];
        soa_product.m_rdata[PIdx::uz][idx_product_start + 1] = soa_product.m_rdata[PIdx::uz][idx_product_start];
    }

}


#endif // WARPX_RECOMBINATION_COLLISION_FUNC_H_