using ITensorMPS
using ITensors
using HDF5
using MKL
include("D_4_hilbertSpace_0.jl")

function get_Hmpo(N::Int64, g::Float64, penalty::Float64)

  os = OpSum() #opsum for the electric field Hamiltonian


  for j in 2:Int(N/2)-1
    os += g^2,"E_u",2*j #electric field
    os += g^2,"E_d",2*j-1 #electric field

  end
  os += g^2,"E_dl",1
  os += g^2,"E_ul",2
  os += g^2,"E_dr",N-1
  os += g^2,"E_ur",N

  #loop for the plaquette terms
  liste = [0,1,2,3]


  for nn in 1:Int(N/2)-1

    for (i, j, k, l) in Iterators.product(0:3, 0:3, 0:3, 0:3)

      i_c = liste[i+1]
      j_c = liste[j+1]
      k_c = liste[k+1]
      l_c = liste[l+1]


      #plaquette_charge_zero
      os += -1/g^2, "corner_"*string(1)*"_"*string(i_c)*"_"*string(j),2*nn-1, "corner_"*string(2)*"_"*string(i)*"_"*string(l), 2*nn, "corner_"*string(3)*"_"*string(j_c)*"_"*string(k),2*nn+1,"corner_"*string(4)*"_"*string(l_c)*"_"*string(k_c),2*nn+2

    end
  end



  #penalty for Abelian selection violation

  for nn in 2:Int(N/2)-2


    os += -2*penalty, "D_R_u",2*nn, "D_L_u", 2*nn+2 #upper right connection
    os += -2*penalty, "D_C_conj",2*nn-1, "D_C", 2*nn #central connection
    os += -2*penalty, "D_R_d",2*nn-1, "D_L_d", 2*nn+1 #lower right connection


    os += penalty, "D_R_u_2",2*nn, "D_L_u_2", 2*nn+2 #upper right connection
    os += penalty, "D_C_conj_2",2*nn-1, "D_C_2", 2*nn #central connection
    os += penalty, "D_R_d_2",2*nn-1, "D_L_d_2", 2*nn+1 #lower right connection



  end



  os += -2*penalty, "D_R",2, "D_L_u", 4 #upper right connection
  os += -2*penalty, "D_C_conj_l",1, "D_C_l", 2 #central connection
  os += -2*penalty, "D_R",1, "D_L_d", 3 #lower right connection


  os += penalty, "D_R_2",2, "D_L_u_2", 4 #upper right connection
  os += penalty, "D_C_conj_l_2",1, "D_C_l_2", 2 #central connection
  os += penalty, "D_R_2",1, "D_L_d_2", 3 #lower right connection



  #horizontals
  os += -2*penalty, "D_R_u",N-2, "D_L", N #upper right connection
  os += -2*penalty, "D_R_d",N-3, "D_L", N-1 #lower right connection

  os += penalty, "D_R_u_2",N-2, "D_L_2", N #upper right connection
  os += penalty, "D_R_d_2",N-3, "D_L_2", N-1 #lower right connection

  #verticals

  os += -2*penalty, "D_C_conj_r",N-1, "D_C_r", N #upper right connection
  os += -2*penalty, "D_C_conj",N-3, "D_C", N-2 #lower right connection


  os += penalty, "D_C_conj_r_2",N-1, "D_C_r_2", N #upper right connection
  os += penalty, "D_C_conj_2",N-3, "D_C_2", N-2 #lower right connection

  os += (N-2+N/2)*penalty, "Id",1


  # os = OpSum() #opsum for the electric field Hamiltonian


  # for j in 2:Int(N/2)-1
  #   os += g^2,"E_u",2*j #electric field
  #   os += g^2,"E_d",2*j-1 #electric field

  # end
  # os += g^2,"E_dl",1
  # os += g^2,"E_ul",2
  # os += g^2,"E_dr",N-1
  # os += g^2,"E_ur",N

  # #loop for the plaquette terms

  # liste = [0,1,2,3]


  # for nn in 1:Int(N/2)-1

  #   for (i, j, k, l) in Iterators.product(0:2, 0:2, 0:2, 0:2)

  #     i_c = liste[i+1]
  #     j_c = liste[j+1]
  #     k_c = liste[k+1]
  #     l_c = liste[l+1]



  #     os += -1/g^2, "corner_"*string(1)*"_"*string(i_c)*"_"*string(j),2*nn-1, "corner_"*string(2)*"_"*string(i)*"_"*string(l), 2*nn, "corner_"*string(3)*"_"*string(j_c)*"_"*string(k),2*nn+1,"corner_"*string(4)*"_"*string(l_c)*"_"*string(k_c),2*nn+2 #plaquette_charge_zero

  #   end
  # end



  # #penalty for Abelian selection violation

  # for nn in 2:Int(N/2)-2


  #   os += -2*penalty, "D_R_u",2*nn, "D_L_u", 2*nn+2 #upper right connection
  #   os += -2*penalty, "D_C_conj",2*nn-1, "D_C", 2*nn #central connection
  #   os += -2*penalty, "D_R_d",2*nn-1, "D_L_d", 2*nn+1 #lower right connection


  #   os += penalty, "D_R_u_2",2*nn, "D_L_u_2", 2*nn+2 #upper right connection
  #   os += penalty, "D_C_conj_2",2*nn-1, "D_C_2", 2*nn #central connection
  #   os += penalty, "D_R_d_2",2*nn-1, "D_L_d_2", 2*nn+1 #lower right connection



  # end



  # os += -2*penalty, "D_R",2, "D_L_u", 4 #upper right connection
  # os += -2*penalty, "D_C_conj_l",1, "D_C_l", 2 #central connection
  # os += -2*penalty, "D_R",1, "D_L_d", 3 #lower right connection


  # os += penalty, "D_R_2",2, "D_L_u_2", 4 #upper right connection
  # os += penalty, "D_C_conj_l_2",1, "D_C_l_2", 2 #central connection
  # os += penalty, "D_R_2",1, "D_L_d_2", 3 #lower right connection



  # #horizontals
  # os += -2*penalty, "D_R_u",N-2, "D_L", N #upper right connection
  # os += -2*penalty, "D_R_d",N-3, "D_L", N-1 #lower right connection

  # os += penalty, "D_R_u_2",N-2, "D_L_2", N #upper right connection
  # os += penalty, "D_R_d_2",N-3, "D_L_2", N-1 #lower right connection

  # #verticals

  # os += -2*penalty, "D_C_conj_r",N-1, "D_C_r", N #upper right connection
  # os += -2*penalty, "D_C_conj",N-3, "D_C", N-2 #lower right connection


  # os += penalty, "D_C_conj_r_2",N-1, "D_C_r_2", N #upper right connection
  # os += penalty, "D_C_conj_2",N-3, "D_C_2", N-2 #lower right connection

  # os += (N-2+N/2)*penalty, "Id",1

  return os 
end

function main(N::Int64,g::Float64,penalty::Float64,D_max::Int64)

  #N_C = 18

  list_of_couplings = [3.,2.,1.5,1.,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.01]
  index_g = findfirst(==(g), list_of_couplings)

  for g in list_of_couplings[index_g:end]
    #rescale constants
    
    #rescale penalty term
    
    max_val = max(g^2, 1/g^2)
    
    
    penalty = penalty*max_val
    
    sites = siteinds("charge_0",N) #dressed sites with zero charge
    
    #redefine sites for boundaries
    sites[1] = siteind("charge_0_b",1) #lower left boundary
    sites[2] = siteind("charge_0_b",2) #upper left boundary
    sites[N-1] = siteind("charge_0_b",N-1) #lower right boundary
    sites[N] = siteind("charge_0_b",N) #upper right boundary
    
    
    if g!=3.
    
      f = h5open("states_dir/D4_ground_zero_charge_"*string(N)*"_"*string(index_g-1)*".h5","r")
    
      sites = read(f,"sites",Vector{Index{Int64}})
    
      close(f)
    
    end
    
    os = get_Hmpo(N,g,penalty)
    H = MPO(os, sites)
    
    # Plan to do nsweep DMRG sweeps:
    nsweeps = 250
    # Set maximum MPS bond dimensions for each sweep
    #i_max = Int(D_max/20)
    #maxdim = [Int(i*D_max/i_max) for i in 1:i_max]
    maxdim = D_max
    # Set maximum truncation error allowed when adapting bond dimensions
    cutoff = [1E-6,1E-9,1E-11]
    etol = 1E-6
    #obs = DemoObserver(etol)
    
    MyObserver = DMRGObserver(;minsweeps = 20, energy_tol = 1E-6)
    
    
    noise = [1E-3,1E-4,1E-5,1E-6,1E-7, 1e-9, 1e-11, 0] #noise for the observer
    
    # Create an initial random matrix product state
    #psi0 = randomMPS(sites, D_init)
    
    #find ground state of infinite couolng in d4
    states_init = [2,2]
    
    
    for i in 1:Int(N/2)-2
      push!(states_init, 13) #add the lower state
      push!(states_init, 13) #add the upper state
    end
    
    push!(states_init, 2) #add the lower edge state
    push!(states_init, 2) #add the upper edge state
    
    psi_init = productMPS(sites, states_init)
    
    
    
    
    # Run the DMRG algorithm, returning energy and optimized MPS
    
    
    
    
    if g!=3.
    
    
    
    
      f = h5open("states_dir/D4_ground_zero_charge_"*string(N)*"_"*string(index_g-1)*".h5","r")
    
    
      psi_init = read(f,"psi_ground",MPS)
    
      close(f)
    
    end
    
    ff = h5open("states_dir/D4_ground_zero_charge_"*string(N)*"_"*string(index_g)*".h5","w")
    
    write(ff,"sites",sites)
    
    
    
    
    energy, psi = dmrg(H, psi_init; nsweeps, observer = MyObserver,noise = noise, maxdim, cutoff)
    write(ff,"psi_ground",psi)
    
    
    close(ff)
    
    
    expect_values_e = zeros(Float64, N)
    
    
    
    for i in 2:Int(N/2)-1
    
      os = OpSum()
      os += 1,"E_u",2*i
      E_u = MPO(os, sites)
      expect_values_e[2*i] = real(inner(psi,Apply(E_u, psi)))
      os = OpSum()
      os += 1,"E_d",2*i-1
      E_d = MPO(os, sites)
      expect_values_e[2*i-1] = real(inner(psi,Apply( E_d, psi)))
    
    end
    
    os = OpSum()
    os += 1,"E_dl",1
    E_dl = MPO(os, sites)
    expect_values_e[1] = real(inner(psi,Apply( E_dl, psi)))
    os = OpSum()
    os += 1,"E_ul",2
    E_ul = MPO(os, sites)
    expect_values_e[2] = real(inner(psi,Apply( E_ul, psi)))
    os = OpSum()
    os += 1,"E_dr",N-1
    E_dr = MPO(os, sites)
    expect_values_e[N-1] = real(inner(psi,Apply( E_dr, psi)))
    os = OpSum()
    os += 1,"E_ur",N
    E_ur = MPO(os, sites)
    expect_values_e[N] = real(inner(psi,Apply( E_ur, psi)))
    
    
    
    res=string(N,"  ", g,"  ",D_max,"  ",energy,"  ",expect_values_e,"\n")
    
    open("results/results_0_N$N", "a") do io
        write(io, res)
    end
    
    
    
    psi = nothing
    H = nothing

    index_g+=1
  end


  







end



main(parse(Int64,ARGS[1]),parse(Float64,ARGS[2]),parse(Float64,ARGS[3]),parse(Int64,ARGS[4]))
