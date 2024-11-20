#ifndef WARPX_IONIZATION_COLLISION_FUNC_H_
#define WARPX_IONIZATION_COLLISION_FUNC_H_

#include "Particles/Collision/CollisionBase.H"
#include "Utils/Parser/ParserUtils.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/ParticleUtils.H"
#include "Particles/Collision/BinaryCollision/NuclearFusion/TwoProductFusionUtil.H"
#include "ablastr/constant.H"
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>




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
    void IonizationInitializeMomentum (
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

        auto& ux1 = soa_1.m_rdata[PIdx::ux][idx_1];
        auto& uy1 = soa_1.m_rdata[PIdx::uy][idx_1];
        auto& uz1 = soa_1.m_rdata[PIdx::uz][idx_1];
        auto& ux2 = soa_2.m_rdata[PIdx::ux][idx_2];
        auto& uy2 = soa_2.m_rdata[PIdx::uy][idx_2];
        auto& uz2 = soa_2.m_rdata[PIdx::uz][idx_2];

        index_type idx_prod_unchanged = idx_prod_0;
        index_type idx_prod_a = idx_prod_1;
        index_type idx_a = 1;

        if (idx_unchanged_product==1){
            idx_prod_unchanged = idx_prod_1;
            idx_prod_a = idx_prod_0;

            idx_a = 0;

        } 
        amrex::ParticleReal mtemp = product_mass[idx_a] + product_mass[idx_unchanged_product];
        
        auto& uxprod0 = soa_products[idx_unchanged_product].m_rdata[PIdx::ux][idx_prod_unchanged];
        auto& uyprod0 = soa_products[idx_unchanged_product].m_rdata[PIdx::uy][idx_prod_unchanged];
        auto& uzprod0 = soa_products[idx_unchanged_product].m_rdata[PIdx::uz][idx_prod_unchanged];

        auto& uxprode = soa_products[idx_unchanged_product].m_rdata[PIdx::ux][idx_prod_unchanged + 1];
        auto& uyprode = soa_products[idx_unchanged_product].m_rdata[PIdx::uy][idx_prod_unchanged + 1];
        auto& uzprode = soa_products[idx_unchanged_product].m_rdata[PIdx::uz][idx_prod_unchanged + 1];

        auto& uxproda = soa_products[idx_a].m_rdata[PIdx::ux][idx_prod_a];
        auto& uyproda = soa_products[idx_a].m_rdata[PIdx::uy][idx_prod_a];
        auto& uzproda = soa_products[idx_a].m_rdata[PIdx::uz][idx_prod_a];



        amrex::ParticleReal uxtempprod, uytempprod, uztempprod;
        
        

        // const amrex::ParticleReal u_coll2 = (ux1-ux2)*(ux1-ux2)+(uy1-uy2)*(uy1-uy2)+(uz1-uz2)*(uz1-uz2);

        // amrex::ParticleReal gamma;
        // amrex::ParticleReal E_coll;
        // ParticleUtils::getCollisionEnergy(u_coll2, m1, m2, gamma, E_coll);

        // auto E_out = static_cast<amrex::ParticleReal>((E_coll - energy_cost) * PhysConst::q_e);

        TwoProductFusionComputeProductMomenta(ux1,uy1,uz1,m1,ux2,uy2,uz2,m2,
                                              uxprod0,uyprod0,uzprod0,product_mass[idx_unchanged_product],
                                              uxtempprod,uytempprod,uztempprod,mtemp,-energy_cost,engine
                                              );
        
        
        // split remaining energy and momentum between the remaining particles
        amrex::ParticleReal Etemp;
        amrex::ParticleReal u2temp = (uxtempprod * uxtempprod) + (uytempprod * uytempprod) + (uztempprod * uztempprod);
        ParticleUtils::getEnergy(u2temp,mtemp,Etemp);
        // energy = mass * u2 / (sqrt(1.0_rt + u2 / c2) + 1.0_rt) / PhysConst::q_e;

        // energy / mass = u2 / (sqrt( 1 + u2/c2) + 1)
        // u2 = e/m *(2*c2 + e/m)/c2

        amrex::ParticleReal E_a = Etemp / 2.0_prt * PhysConst::q_e / product_mass[idx_a]; // actually E/m 
        
        amrex::ParticleReal E_c = Etemp / 2.0_prt * PhysConst::q_e / product_mass[idx_unchanged_product];

        

        amrex::ParticleReal uproda =  sqrt((2.0_prt* c2 * E_a) + (E_a*E_a))/PhysConst::c;
        
        amrex::ParticleReal uprodc =  sqrt((2.0_prt* c2 * E_c) + (E_c*E_c))/PhysConst::c;


        ParticleUtils::RandomizeVelocity(uxproda,uyproda,uzproda,uproda,engine);
        
        ParticleUtils::RandomizeVelocity(uxprode,uyprode,uzprode,uprodc,engine);

        // Force both this index and the next to have the same momentum
        // This is because DSMC creates two particles
        soa_products[idx_unchanged_product].m_rdata[PIdx::ux][idx_prod_unchanged + 2] = uxprod0;
        soa_products[idx_unchanged_product].m_rdata[PIdx::uy][idx_prod_unchanged + 2] = uyprod0;
        soa_products[idx_unchanged_product].m_rdata[PIdx::uz][idx_prod_unchanged + 2] = uzprod0;

        // These two can be randomized separately
        auto& uxprode2 = soa_products[idx_unchanged_product].m_rdata[PIdx::ux][idx_prod_unchanged + 3];
        auto& uyprode2 = soa_products[idx_unchanged_product].m_rdata[PIdx::uy][idx_prod_unchanged + 3];
        auto& uzprode2 = soa_products[idx_unchanged_product].m_rdata[PIdx::uz][idx_prod_unchanged + 3];
        auto& uxproda2 = soa_products[idx_a].m_rdata[PIdx::ux][idx_prod_a + 1];
        auto& uyproda2 = soa_products[idx_a].m_rdata[PIdx::uy][idx_prod_a + 1];
        auto& uzproda2 = soa_products[idx_a].m_rdata[PIdx::uz][idx_prod_a + 1];


        ParticleUtils::RandomizeVelocity(uxproda2,uyproda2,uzproda2,uproda,engine);
        
        ParticleUtils::RandomizeVelocity(uxprode2,uyprode2,uzprode2,uprodc,engine);

    }


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
    void IonizationInitializeMomentum (
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

        auto& ux1 = soa_1.m_rdata[PIdx::ux][idx_1];
        auto& uy1 = soa_1.m_rdata[PIdx::uy][idx_1];
        auto& uz1 = soa_1.m_rdata[PIdx::uz][idx_1];
        auto& ux2 = soa_2.m_rdata[PIdx::ux][idx_2];
        auto& uy2 = soa_2.m_rdata[PIdx::uy][idx_2];
        auto& uz2 = soa_2.m_rdata[PIdx::uz][idx_2];

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
        amrex::ParticleReal mtemp = product_mass[idx_a] + product_mass[idx_b] + product_mass[idx_unchanged_product];
        
        auto& uxprod0 = soa_products[idx_unchanged_product].m_rdata[PIdx::ux][idx_prod_unchanged];
        auto& uyprod0 = soa_products[idx_unchanged_product].m_rdata[PIdx::uy][idx_prod_unchanged];
        auto& uzprod0 = soa_products[idx_unchanged_product].m_rdata[PIdx::uz][idx_prod_unchanged];

        auto& uxprode = soa_products[idx_unchanged_product].m_rdata[PIdx::ux][idx_prod_unchanged + 1];
        auto& uyprode = soa_products[idx_unchanged_product].m_rdata[PIdx::uy][idx_prod_unchanged + 1];
        auto& uzprode = soa_products[idx_unchanged_product].m_rdata[PIdx::uz][idx_prod_unchanged + 1];

        auto& uxproda = soa_products[idx_a].m_rdata[PIdx::ux][idx_prod_a];
        auto& uyproda = soa_products[idx_a].m_rdata[PIdx::uy][idx_prod_a];
        auto& uzproda = soa_products[idx_a].m_rdata[PIdx::uz][idx_prod_a];
        auto& uxprodb = soa_products[idx_b].m_rdata[PIdx::ux][idx_prod_b];
        auto& uyprodb = soa_products[idx_b].m_rdata[PIdx::uy][idx_prod_b];
        auto& uzprodb = soa_products[idx_b].m_rdata[PIdx::uz][idx_prod_b];


        amrex::ParticleReal uxtempprod, uytempprod, uztempprod;
        
        

        // const amrex::ParticleReal u_coll2 = (ux1-ux2)*(ux1-ux2)+(uy1-uy2)*(uy1-uy2)+(uz1-uz2)*(uz1-uz2);

        // amrex::ParticleReal gamma;
        // amrex::ParticleReal E_coll;
        // ParticleUtils::getCollisionEnergy(u_coll2, m1, m2, gamma, E_coll);

        // auto E_out = static_cast<amrex::ParticleReal>((E_coll - energy_cost) * PhysConst::q_e);

        TwoProductFusionComputeProductMomenta(ux1,uy1,uz1,m1,ux2,uy2,uz2,m2,
                                              uxprod0,uyprod0,uzprod0,product_mass[idx_unchanged_product],
                                              uxtempprod,uytempprod,uztempprod,mtemp,-energy_cost,engine
                                              );
        
        
        // split remaining energy and momentum between the remaining particles
        amrex::ParticleReal Etemp;
        amrex::ParticleReal u2temp = (uxtempprod * uxtempprod) + (uytempprod * uytempprod) + (uztempprod * uztempprod);
        ParticleUtils::getEnergy(u2temp,mtemp,Etemp);
        // energy = mass * u2 / (sqrt(1.0_rt + u2 / c2) + 1.0_rt) / PhysConst::q_e;

        // energy / mass = u2 / (sqrt( 1 + u2/c2) + 1)
        // u2 = e/m *(2*c2 + e/m)/c2

        amrex::ParticleReal E_a = Etemp / 3.0_prt * PhysConst::q_e / product_mass[idx_a]; // actually E/m 
        amrex::ParticleReal E_b = Etemp / 3.0_prt * PhysConst::q_e / product_mass[idx_b];
        amrex::ParticleReal E_c = Etemp / 3.0_prt * PhysConst::q_e / product_mass[idx_unchanged_product];

        

        amrex::ParticleReal uproda =  sqrt((2.0_prt* c2 * E_a) + (E_a*E_a))/PhysConst::c;
        amrex::ParticleReal uprodb =  sqrt((2.0_prt* c2 * E_b) + (E_b*E_b))/PhysConst::c;
        amrex::ParticleReal uprodc =  sqrt((2.0_prt* c2 * E_c) + (E_c*E_c))/PhysConst::c;


        ParticleUtils::RandomizeVelocity(uxproda,uyproda,uzproda,uproda,engine);
        ParticleUtils::RandomizeVelocity(uxprodb,uyprodb,uzprodb,uprodb,engine);
        ParticleUtils::RandomizeVelocity(uxprode,uyprode,uzprode,uprodc,engine);


        // Force both this index and the next to have the same momentum
        // This is because DSMC creates two particles
        soa_products[idx_unchanged_product].m_rdata[PIdx::ux][idx_prod_unchanged + 2] = uxprod0;
        soa_products[idx_unchanged_product].m_rdata[PIdx::uy][idx_prod_unchanged + 2] = uyprod0;
        soa_products[idx_unchanged_product].m_rdata[PIdx::uz][idx_prod_unchanged + 2] = uzprod0;

        // These two can be randomized separately
        auto& uxprode2 = soa_products[idx_unchanged_product].m_rdata[PIdx::ux][idx_prod_unchanged + 3];
        auto& uyprode2 = soa_products[idx_unchanged_product].m_rdata[PIdx::uy][idx_prod_unchanged + 3];
        auto& uzprode2 = soa_products[idx_unchanged_product].m_rdata[PIdx::uz][idx_prod_unchanged + 3];
        auto& uxproda2 = soa_products[idx_a].m_rdata[PIdx::ux][idx_prod_a + 1];
        auto& uyproda2 = soa_products[idx_a].m_rdata[PIdx::uy][idx_prod_a + 1];
        auto& uzproda2 = soa_products[idx_a].m_rdata[PIdx::uz][idx_prod_a + 1];
        auto& uxprodb2 = soa_products[idx_b].m_rdata[PIdx::ux][idx_prod_b + 1];
        auto& uyprodb2 = soa_products[idx_b].m_rdata[PIdx::uy][idx_prod_b + 1];
        auto& uzprodb2 = soa_products[idx_b].m_rdata[PIdx::uz][idx_prod_b + 1];
    

        ParticleUtils::RandomizeVelocity(uxproda2,uyproda2,uzproda2,uproda,engine);
        ParticleUtils::RandomizeVelocity(uxprodb2,uyprodb2,uzprodb2,uprodb,engine);
        ParticleUtils::RandomizeVelocity(uxprode2,uyprode2,uzprode2,uprodc,engine);
        


    }

}


#endif // WARPX_DISSOCIATION_COLLISION_FUNC_H_