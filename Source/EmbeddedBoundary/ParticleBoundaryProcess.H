/* Copyright 2021 Andrew Myers
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLEBOUNDARYPROCESS_H_
#define WARPX_PARTICLEBOUNDARYPROCESS_H_

#include <AMReX_Particle.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>
#include <AMReX_Random.H>


namespace ParticleBoundaryProcess {

struct NoOp {
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (const PData& /*ptd*/, int /*i*/,
                     const amrex::RealVect& /*pos*/, const amrex::RealVect& /*normal*/,
                    const amrex::RealVect & /*vel*/,
                     amrex::RandomEngine const& /*engine*/) const noexcept
    {}
};

struct Absorb {
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (PData& ptd, int i,
                     const amrex::RealVect& /*pos*/, const amrex::RealVect& /*normal*/,
                     const amrex::RealVect & /*vel*/,
                     amrex::RandomEngine const& /*engine*/) const noexcept
    {
        amrex::ParticleIDWrapper{ptd.m_idcpu[i]}.make_invalid();
    }
};
struct Reflect {
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (PData& ptd, int i,
                     const amrex::RealVect& pos /*pos*/, const amrex::RealVect& n /*normal*/, const amrex::RealVect & u /*vel*/,
                     amrex::RandomEngine const& /*engine*/) const noexcept
    {
        //amrex::ParticleIDWrapper{ptd.m_idcpu[i]}.make_invalid();
        /*#STEP 2: use these parameters to inject particle from the same position in the plasma
            elect_pc = particle_containers.ParticleContainerWrapper(name) #general particle container
            

            ####this part is specific to the case of simple reflection.
            un=ux*nx+uy*ny+uz*nz
            ux_reflect=-2*un*nx+ux #for a "mirror reflection" u(sym)=-2(u.n)n+u
            uy_reflect=-2*un*ny+uy
            uz_reflect=-2*un*nz+uz
            
            elect_pc.add_particles(
                x=x + (dtI-delta_t)*ux_reflect, y=y + (dtI-delta_t)*uy_reflect, z=z + (dtI-delta_t)*uz_reflect,
                ux=ux_reflect, uy=uy_reflect, uz=uz_reflect,
                w=w
                ) #adds the particle in the general particle container at the next step
                #### Can be modified depending on the model of interaction.*/
        // TODO: Finish this. I currently need to get the time difference in here. 
        // I probably can follow the ParticleBoundaryBuffer routine
    }
};
}

#endif //WARPX_PARTICLEBOUNDARYPROCESS_H_
