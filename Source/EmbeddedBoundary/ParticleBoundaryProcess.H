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
    void operator() (PData& ptd, int index,
                     const amrex::RealVect& pos /*pos*/, const amrex::RealVect& n /*normal*/, const amrex::RealVect & u /*vel*/,
                     amrex::RandomEngine const& /*engine*/) const noexcept
    {
        /*
        // Modify the position of the destination particle:
        // Move it to the point of intersection with the embedded boundary
        // (which is found by using a bisection algorithm)
        auto phiarr = (*distance_to_eb[ptd.GetLevel()])[index].array();  // signed distance function
        auto &warpx = WarpX::GetInstance();
        const auto dt = warpx.getdt(ptd.GetLevel());
        const auto& p = ptd.getSuperParticle(index);
        amrex::ParticleReal xp, yp, zp;
        get_particle_position( p, xp, yp, zp );
        amrex::ParticleReal const ux = ptd.m_rdata[PIdx::ux][index];
        amrex::ParticleReal const uy = ptd.m_rdata[PIdx::uy][index];
        amrex::ParticleReal const uz = ptd.m_rdata[PIdx::uz][index];

        
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dxi = m_dxi;
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const plo = m_plo;

        // Bisection algorithm to find the point where phi(x,y,z)=0 (i.e. on the embedded boundary)
        amrex::Real const dt_fraction = amrex::bisect( 0.0, 1.0,
            [=] (amrex::Real dt_frac) {
                int i, j, k;
                amrex::Real W[AMREX_SPACEDIM][2];
                amrex::ParticleReal x_temp=xp, y_temp=yp, z_temp=zp;
                UpdatePosition(x_temp, y_temp, z_temp, ux, uy, uz, -dt_frac*dt);
                ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                    x_temp, y_temp, z_temp, plo, dxi, i, j, k, W);
                amrex::Real const phi_value = ablastr::particles::interp_field_nodal(i, j, k, W, phiarr);
                return phi_value;
            } );

        // Also record the real time on the destination
        ptd.m_runtime_idata[m_step_index][index] = m_step;
        auto delta_t = (1._rt- dt_fraction)*m_dt;

        // Now that dt_fraction has be obtained (with bisect)
        // Save the corresponding position of the particle at the boundary
        amrex::ParticleReal x_temp=xp, y_temp=yp, z_temp=zp;
        UpdatePosition(x_temp, y_temp, z_temp, ux, uy, uz, -dt_fraction*m_dt);

        // record the components of the normal on the destination
        int i, j, k;
        amrex::Real W[AMREX_SPACEDIM][2];
        ablastr::particles::compute_weights<amrex::IndexType::NODE>(
            x_temp, y_temp, z_temp, plo, dxi, i, j, k, W);
        int ic, jc, kc; // Cell-centered indices
        amrex::Real Wc[AMREX_SPACEDIM][2]; // Cell-centered weight
        ablastr::particles::compute_weights<amrex::IndexType::CELL>(
            x_temp, y_temp, z_temp, plo, dxi, ic, jc, kc, Wc);
        amrex::RealVect normal = DistanceToEB::interp_normal(i, j, k, W, ic, jc, kc, Wc, phiarr, dxi);
        DistanceToEB::normalize(normal);

#if (defined WARPX_DIM_3D)
        ptd.m_rdata[PIdx::x][index] = x_temp;
        ptd.m_rdata[PIdx::y][index] = y_temp;
        ptd.m_rdata[PIdx::z][index] = z_temp;
        //save normal components
        ptd.m_runtime_rdata[m_normal_index][index] = normal[0];
        ptd.m_runtime_rdata[m_normal_index+1][index] = normal[1];
        ptd.m_runtime_rdata[m_normal_index+2][index] = normal[2];
#elif (defined WARPX_DIM_XZ)
        ptd.m_rdata[PIdx::x][index] = x_temp;
        ptd.m_rdata[PIdx::z][index] = z_temp;
        amrex::ignore_unused(y_temp);
        //save normal components
        ptd.m_runtime_rdata[m_normal_index][index] = normal[0];
        ptd.m_runtime_rdata[m_normal_index+1][index] = 0.0;
        ptd.m_runtime_rdata[m_normal_index+2][index] = normal[1];
#elif (defined WARPX_DIM_RZ)
        ptd.m_rdata[PIdx::x][index] = std::sqrt(x_temp*x_temp + y_temp*y_temp);
        ptd.m_rdata[PIdx::z][index] = z_temp;
        ptd.m_rdata[PIdx::theta][index] = std::atan2(y_temp, x_temp);
        //save normal components
        amrex::Real const theta = std::atan2(y_temp, x_temp);
        ptd.m_runtime_rdata[m_normal_index][index] = normal[0]*std::cos(theta);
        ptd.m_runtime_rdata[m_normal_index+1][index] = normal[0]*std::sin(theta);
        ptd.m_runtime_rdata[m_normal_index+2][index] = normal[1];
#elif (defined WARPX_DIM_1D_Z)
        ptd.m_rdata[PIdx::z][index] = z_temp;
        amrex::ignore_unused(x_temp, y_temp);
        //normal not defined
        ptd.m_runtime_rdata[m_normal_index][index] = 0.0;
        ptd.m_runtime_rdata[m_normal_index+1][index] = 0.0;
        ptd.m_runtime_rdata[m_normal_index+2][index] = 0.0;
#else
        amrex::ignore_unused(x_temp, y_temp, z_temp,normal);
#endif

        // flip id to positive in destination
        amrex::ParticleIDWrapper{ptd.m_idcpu[index]}.make_valid();
    
        //amrex::ParticleIDWrapper{ptd.m_idcpu[i]}.make_invalid();
        */

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
