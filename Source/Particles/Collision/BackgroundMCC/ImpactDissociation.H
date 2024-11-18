#ifndef WARPX_PARTICLES_COLLISION_IMPACT_DISSOCIATION_H_
#define WARPX_PARTICLES_COLLISION_IMPACT_DISSOCIATION_H_

#include "Particles/Collision/ScatteringProcess.H"

#include "Particles/NamedComponentParticleContainer.H"
#include "Utils/ParticleUtils.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Random.H>
#include <AMReX_REAL.H>



/**
 * \brief Transform functor for impact dissociation
 */
class ImpactDissociationTransformFunc
{
public:

    /**
    * \brief Constructor of the ImpactDissociationTransformFunc functor.
    *
    * The transform is responsible for appropriately decreasing the kinetic
    * energy of the colliding particle and assigning appropriate velocities
    * to the two newly created particles. To this end the energy cost of
    * ionization is passed to the constructor as well as the mass of the
    * colliding species and the standard deviation of the ion velocity
    * (normalized temperature).
    * Note that the mass and energy quantities have to be stored as doubles to
    * ensure accurate energy calculations (otherwise errors occur with single
    * or mixed precision builds of WarpX).
    *
    * @param[in] energy_cost energy cost of ionization
    * @param[in] mass1 mass of the colliding species
    * @param[in] sqrt_kb_m value of sqrt(kB/m), where kB is Boltzmann's constant
                 and m is the background neutral mass
    * @param[in] T_a_func ParserExecutor<4> function to get the background
                 temperature in Kelvin as a function of space and time
    * @param[in] t the current simulation time
    */
    ImpactDissociationTransformFunc(
        amrex::ParticleReal energy_cost, double mass1, amrex::ParticleReal sqrt_kb_m,
        amrex::ParserExecutor<4> const& T_a_func, amrex::Real t
    ) :  m_energy_cost(energy_cost), m_mass1(mass1),
         m_sqrt_kb_m(sqrt_kb_m), m_T_a_func(T_a_func), m_t(t) { }

    /**
    * \brief Functor call. It determines the properties of the generated pair
    * and decreases the kinetic energy of the colliding particle. Inputs are
    * standard from the FilterCopyTransfrom::filterCopyTransformParticles
    * function.
    *
    * @param[in,out] dst1 target species 1 (electrons)
    * @param[in] src source species (electrons)
    * @param[in] i_src particle index of the source species
    * @param[in] i_dst1 particle index of target species 1
    * @param[in] engine random number generator engine
    */
    template <typename DstData, typename SrcData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (DstData& dst1, SrcData& src,
        int const i_src, int const i_dst1,
        amrex::RandomEngine const& engine) const noexcept
    {
        using namespace amrex;
        using std::sqrt;

        // get references to the particle to get its position
        const auto& p = src.getSuperParticle(i_src);
        ParticleReal x, y, z;
        double E_coll;
        get_particle_position(p, x, y, z);

        // calculate standard deviation in neutral velocity distribution using
        // the local temperature
        // Divide by two since this velocity will be shared by the two neutrals
        const ParticleReal bg_vel_std = m_sqrt_kb_m * std::sqrt(m_T_a_func(x, y, z, m_t));

        // get references to the original particle's velocity
        auto& ux = src.m_rdata[PIdx::ux][i_src];
        auto& uy = src.m_rdata[PIdx::uy][i_src];
        auto& uz = src.m_rdata[PIdx::uz][i_src];

        // get references to the new particles' velocities
        auto& n1_ux = dst1.m_rdata[PIdx::ux][i_dst1];
        auto& n1_uy = dst1.m_rdata[PIdx::uy][i_dst1];
        auto& n1_uz = dst1.m_rdata[PIdx::uz][i_dst1];
        auto& n2_ux = dst1.m_rdata[PIdx::ux][i_dst1+1];
        auto& n2_uy = dst1.m_rdata[PIdx::uy][i_dst1+1];
        auto& n2_uz = dst1.m_rdata[PIdx::uz][i_dst1+1];


        // NOTE that this calculation assumes that the background energy is small compared to the projectile
        // calculate kinetic energy
        const ParticleReal u_coll2 = ux*ux + uy*uy + uz*uz;
        ParticleUtils::getEnergy(u_coll2, m_mass1, E_coll);
        

        // electron only loses the penalty cost
        // each electron gets half the energy (could change this later)
        const auto E_out = static_cast<amrex::ParticleReal>((E_coll - m_energy_cost) * PhysConst::q_e);

        // precalculate often used value
        constexpr auto c2 = PhysConst::c * PhysConst::c;
        const auto mc2 = m_mass1*c2;

        const amrex::ParticleReal up = sqrt(E_out * (E_out + 2.0_prt*mc2) / c2) / m_mass1;

        // isotropically scatter electrons
        ParticleUtils::RandomizeVelocity(ux, uy, uz, up, engine);
        

        // get velocities for the neutrals from a Maxwellian distribution
        n1_ux = bg_vel_std * RandomNormal(0_prt, 1.0_prt, engine);
        n1_uy = bg_vel_std * RandomNormal(0_prt, 1.0_prt, engine);
        n1_uz = bg_vel_std * RandomNormal(0_prt, 1.0_prt, engine);
        n2_ux = bg_vel_std * RandomNormal(0_prt, 1.0_prt, engine);
        n2_uy = bg_vel_std * RandomNormal(0_prt, 1.0_prt, engine);
        n2_uz = bg_vel_std * RandomNormal(0_prt, 1.0_prt, engine);
    }

private:
    amrex::ParticleReal m_energy_cost;
    double m_mass1;
    amrex::ParticleReal m_sqrt_kb_m;
    amrex::ParserExecutor<4> m_T_a_func;
    amrex::Real m_t;
};
#endif // WARPX_PARTICLES_COLLISION_IMPACT_DISSOCIATION_H_
