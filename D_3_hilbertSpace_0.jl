using ITensorMPS
using ITensors
using HDF5
using MKL

const alpha_1 = -1.
const alpha_2 = 10.
const alpha_3 = 0.

const alpha_3_tilde = 1.05


mutable struct DemoObserver <: AbstractObserver
  energy_tol::Float64
  last_energy::Float64

  DemoObserver(energy_tol=0.0) = new(energy_tol,1000.0)
end

function ITensorMPS.checkdone!(o::DemoObserver;kwargs...)
  sw = kwargs[:sweep]
  energy = kwargs[:energy]
  if abs(energy-o.last_energy)/abs(energy) < o.energy_tol
    println("Stopping DMRG after sweep $sw")
    return true
  end
  # Otherwise, update last_energy and keep going
  o.last_energy = energy
  return false
end

function kron_delta(j::Int64, N::Int64) #takes care of periodic boundary conditions
  if j == Int(N/2)-1
    return N
  else 
    return 0
  end
end

ITensors.space(::SiteType"charge_0") = 11 #normal dressed site
ITensors.space(::SiteType"charge_0_b") = 3 #boundary dressed site


#additional functions for spin l

#list_strings_0 = ["D1_","D2_","U1_","U2_"]
#list_strings_b = ["D1_ledge_","D2_ledge_","D1_redge_","D2_redge_","U1_ledge_","U2_ledge_","U1_redge_","U2_redge_"]
#list_strings_1 = ["U1_charge_","U2_charge_"]



for nn in 1:4

  for i in 0:2

    for j in 0:2

      A = OpName{Symbol("corner_"*string(nn)*"_"*string(i)*"_"*string(j))} 
      function ITensors.op!(Op::ITensor, 
                          ::A,
                          ::SiteType"charge_0",
                          s::Index)

        open("corner_operators/corner_"*string(nn-1)*"_"*string(i)*"_"*string(j)*".txt") do f 

          # line_number
          line = 0   
        
          # read till end of file
          while ! eof(f)  
        
              # read a new / next line for every iteration           
              inp = split(readline(f))  # Split line into words/fields

              # Assuming inp has at least 4 fields: key1, key2, real, imag
              key1 = Int(parse(Float64, inp[1]))
              key2 = Int(parse(Float64, inp[2]))
              real = parse(Float64, inp[3])
              imag = parse(Float64, inp[4])
              # Construct the operator using the keys and values
              Op[s'=>key1,s=>key2] = real + imag * 1im

              line += 1
          end
        
        end
                          
      end

    end
  end
end

for nn in [1,2]

  for i in 0:2

    for j in 0:2

      A = OpName{Symbol("corner_"*string(2*nn)*"_"*string(i)*"_"*string(j))} 
      function ITensors.op!(Op::ITensor, 
                          ::A,
                          ::SiteType"charge_0_b",
                          s::Index)

        open("corner_operators/corner_u_edge_0_"*string(nn-1)*"_"*string(i)*"_"*string(j)*".txt") do f 

          # line_number
          line = 0   
        
          # read till end of file
          while ! eof(f)  
        
              # read a new / next line for every iteration           
              inp = split(readline(f))  # Split line into words/fields

              # Assuming inp has at least 4 fields: key1, key2, real, imag
              key1 = Int(parse(Float64, inp[1]))
              key2 = Int(parse(Float64, inp[2]))
              real = parse(Float64, inp[3])
              imag = parse(Float64, inp[4])
              # Construct the operator using the keys and values
              Op[s'=>key1,s=>key2] = real + imag * 1im

              line += 1
          end
        
        end
                          
      end

    end
  end
end

for nn in [1,2]

  for i in 0:2

    for j in 0:2

      A = OpName{Symbol("corner_"*string(2*nn-1)*"_"*string(i)*"_"*string(j))} 
      function ITensors.op!(Op::ITensor, 
                          ::A,
                          ::SiteType"charge_0_b",
                          s::Index)

        open("corner_operators/corner_d_edge_"*string(nn-1)*"_"*string(i)*"_"*string(j)*".txt") do f 

          # line_number
          line = 0   
        
          # read till end of file
          while ! eof(f)  
        
              # read a new / next line for every iteration           
              inp = split(readline(f))  # Split line into words/fields

              # Assuming inp has at least 4 fields: key1, key2, real, imag
              key1 = Int(parse(Float64, inp[1]))
              key2 = Int(parse(Float64, inp[2]))
              real = parse(Float64, inp[3])
              imag = parse(Float64, inp[4])
              # Construct the operator using the keys and values
              Op[s'=>key1,s=>key2] = real + imag * 1im

              line += 1
          end
        
        end
                          
      end

    end
  end
end


#penalty matrices for charge-0 dressed sites

function ITensors.op!(Op::ITensor, 
                     ::OpName"D_R",
                     ::SiteType"charge_0_b",
                     s::Index)

  open("constraints/D_R_ul_0.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real 

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"D_R_2",
                     ::SiteType"charge_0_b",
                     s::Index)

  open("constraints/D_R_ul_0.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real^2 

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"D_L",
                     ::SiteType"charge_0_b",
                     s::Index)

  open("constraints/D_L_ur_0.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real 

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"D_L_2",
                     ::SiteType"charge_0_b",
                     s::Index)

  open("constraints/D_L_ur_0.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real^2 

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"D_C_l",
                     ::SiteType"charge_0_b",
                     s::Index)

  open("constraints/D_C_ul_0.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real 

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"D_C_l_2",
                     ::SiteType"charge_0_b",
                     s::Index)

  open("constraints/D_C_ul_0.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real^2 

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"D_C_r",
                     ::SiteType"charge_0_b",
                     s::Index)

  open("constraints/D_C_ur_0.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real 

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"D_C_r_2",
                     ::SiteType"charge_0_b",
                     s::Index)

  open("constraints/D_C_ur_0.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real^2 

      line += 1
    end
   
  end
                     
end


function ITensors.op!(Op::ITensor, 
                     ::OpName"D_C_conj_l",
                     ::SiteType"charge_0_b",
                     s::Index)

  open("constraints/D_C_dl.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real 

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"D_C_conj_l_2",
                     ::SiteType"charge_0_b",
                     s::Index)

  open("constraints/D_C_dl.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real^2 

      line += 1
    end
   
  end
                     
end


function ITensors.op!(Op::ITensor, 
                     ::OpName"D_C_conj_r",
                     ::SiteType"charge_0_b",
                     s::Index)

  open("constraints/D_C_dr.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real 

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"D_C_conj_r_2",
                     ::SiteType"charge_0_b",
                     s::Index)

  open("constraints/D_C_dr.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real^2 

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"D_R",
                     ::SiteType"charge_0_b",
                     s::Index)

  open("constraints/D_R_dl.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real 

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"D_R_2",
                     ::SiteType"charge_0_b",
                     s::Index)

  open("constraints/D_R_dl.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real^2 

      line += 1
    end
   
  end
                     
end


function ITensors.op!(Op::ITensor, 
                     ::OpName"D_L",
                     ::SiteType"charge_0_b",
                     s::Index)

  open("constraints/D_L_dr.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real 

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"D_L_2",
                     ::SiteType"charge_0_b",
                     s::Index)

  open("constraints/D_L_dr.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real^2 

      line += 1
    end
   
  end
                     
end


function ITensors.op!(Op::ITensor, 
                     ::OpName"D_R_u",
                     ::SiteType"charge_0",
                     s::Index)

  open("constraints/D_R_u.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real 

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"D_R_u_2",
                     ::SiteType"charge_0",
                     s::Index)

  open("constraints/D_R_u.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real^2 

      line += 1
    end
   
  end
                     
end




function ITensors.op!(Op::ITensor, 
                     ::OpName"D_L_u",
                     ::SiteType"charge_0",
                     s::Index)

  open("constraints/D_L_u.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real 

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"D_L_u_2",
                     ::SiteType"charge_0",
                     s::Index)

  open("constraints/D_L_u.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real^2 

      line += 1
    end
   
  end
                     
end


function ITensors.op!(Op::ITensor, 
                     ::OpName"D_R_d",
                     ::SiteType"charge_0",
                     s::Index)

  open("constraints/D_R_d.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real 

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"D_R_d_2",
                     ::SiteType"charge_0",
                     s::Index)

  open("constraints/D_R_d.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real^2 

      line += 1
    end
   
  end
                     
end


function ITensors.op!(Op::ITensor, 
                     ::OpName"D_L_d",
                     ::SiteType"charge_0",
                     s::Index)

  open("constraints/D_L_d.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real 

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"D_L_d_2",
                     ::SiteType"charge_0",
                     s::Index)

  open("constraints/D_L_d.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real^2 

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"D_C",
                     ::SiteType"charge_0",
                     s::Index)

  open("constraints/D_C_u.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real 

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"D_C_2",
                     ::SiteType"charge_0",
                     s::Index)

  open("constraints/D_C_u.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real^2 

      line += 1
    end
   
  end
                     
end


function ITensors.op!(Op::ITensor, 
                     ::OpName"D_C_conj",
                     ::SiteType"charge_0",
                     s::Index)

  open("constraints/D_C_d.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real 

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"D_C_conj_2",
                     ::SiteType"charge_0",
                     s::Index)

  open("constraints/D_C_d.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])
      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real^2 

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"E_u",
                     ::SiteType"charge_0",
                     s::Index)

  open("efields/efield_u.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])

      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"E_d",
                     ::SiteType"charge_0",
                     s::Index)

  open("efields/efield_d.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])

      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real

      line += 1
    end
   
  end
                     
end


function ITensors.op!(Op::ITensor, 
                     ::OpName"E_ul",
                     ::SiteType"charge_0_b",
                     s::Index)

  open("efields/efield_ul_0.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])

      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real

      line += 1
    end
   
  end
                     
end


function ITensors.op!(Op::ITensor, 
                     ::OpName"E_ur",
                     ::SiteType"charge_0_b",
                     s::Index)

  open("efields/efield_ur_0.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])

      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"E_dl",
                     ::SiteType"charge_0_b",
                     s::Index)

  open("efields/efield_dl.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])

      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real

      line += 1
    end
   
  end
                     
end

function ITensors.op!(Op::ITensor, 
                     ::OpName"E_dr",
                     ::SiteType"charge_0_b",
                     s::Index)

  open("efields/efield_dr.txt") do f 

    # line_number
    line = 0   
  
    # read till end of file
    while ! eof(f)  
   
      # read a new / next line for every iteration           
      inp = split(readline(f))  # Split line into words/fields

      # Assuming inp has at least 4 fields: key1, key2, real, imag
      key1 = Int(parse(Float64, inp[1]))
      key2 = Int(parse(Float64, inp[2]))
      real = parse(Float64, inp[3])

      # Construct the operator using the keys and values
      Op[s'=>key1,s=>key2] = real

      line += 1
    end
   
  end
                     
end
