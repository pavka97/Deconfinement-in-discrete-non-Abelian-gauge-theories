using ITensorMPS
using ITensors
using HDF5


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
ITensors.space(::SiteType"charge_1") = 5 #charged dressed site
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
                          ::SiteType"charge_1",
                          s::Index)

        open("corner_operators/corner_u_edge_"*string(nn-1)*"_"*string(i)*"_"*string(j)*".txt") do f 

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
                     ::SiteType"charge_1",
                     s::Index)

  open("constraints/D_R_ul.txt") do f 

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
                     ::SiteType"charge_1",
                     s::Index)

  open("constraints/D_R_ul.txt") do f 

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
                     ::SiteType"charge_1",
                     s::Index)

  open("constraints/D_L_ur.txt") do f 

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
                     ::SiteType"charge_1",
                     s::Index)

  open("constraints/D_L_ur.txt") do f 

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
                     ::SiteType"charge_1",
                     s::Index)

  open("constraints/D_C_ul.txt") do f 

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
                     ::SiteType"charge_1",
                     s::Index)

  open("constraints/D_C_ul.txt") do f 

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
                     ::SiteType"charge_1",
                     s::Index)

  open("constraints/D_C_ur.txt") do f 

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
                     ::SiteType"charge_1",
                     s::Index)

  open("constraints/D_C_ur.txt") do f 

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
                     ::SiteType"charge_1",
                     s::Index)

  open("efields/efield_ul.txt") do f 

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
                     ::SiteType"charge_1",
                     s::Index)

  open("efields/efield_ur.txt") do f 

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



function main(N::Int64,g::Float64,penalty::Float64,D_max::Int64)

  N_C = 18

  list_of_couplings = [3.,2.,1.5,1.,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.01]
  index_g = findfirst(==(g), list_of_couplings)


  #rescale constants



  #rescale penalty term

  max_val = max(g^2, 1/g^2)


  penalty = penalty*max_val





  sites = siteinds("charge_0",N) #dressed sites with zero charge

  #redefine sites for boundaries
  sites[1] = siteind("charge_0_b",1) #lower left boundary
  sites[2] = siteind("charge_1",2) #upper left boundary
  sites[N-1] = siteind("charge_0_b",N-1) #lower right boundary
  sites[N] = siteind("charge_1",N) #upper right boundary


  if g!=3.

    f = h5open("D3_ground_and_excited_"*string(N)*"_"*string(index_g-1)*".h5","r")

    sites = read(f,"sites",Vector{Index{Int64}})

    close(f)

  end




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

  liste = [1,0,2]


  for nn in 1:Int(N/2)-1

    for (i, j, k, l) in Iterators.product(0:2, 0:2, 0:2, 0:2)

      i_c = liste[i+1]
      j_c = liste[j+1]
      k_c = liste[k+1]
      l_c = liste[l+1]



      os += -1/g^2, "corner_"*string(1)*"_"*string(i_c)*"_"*string(j),2*nn-1, "corner_"*string(2)*"_"*string(i)*"_"*string(l), 2*nn, "corner_"*string(3)*"_"*string(j_c)*"_"*string(k),2*nn+1,"corner_"*string(4)*"_"*string(l_c)*"_"*string(k_c),2*nn+2 #plaquette_charge_zero

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


  H = MPO(os, sites)



  # Plan to do nsweep DMRG sweeps:
  nsweeps = 10000
  # Set maximum MPS bond dimensions for each sweep
  #i_max = Int(D_max/20)
  #maxdim = [Int(i*D_max/i_max) for i in 1:i_max]
  maxdim = D_max
  # Set maximum truncation error allowed when adapting bond dimensions
  cutoff = [1E-6,1E-9,1E-11,1e-12]
  etol = 1E-11
  obs = DemoObserver(etol)

  MyObserver = DMRGObserver(;minsweeps = 20, energy_tol = 1E-6)
  MyObserver_1 = DMRGObserver(;minsweeps = 20, energy_tol = 1E-6)


  noise = [1E-3,1E-4,1E-5,1E-6,1E-7] #noise for the observer

  # Create an initial random matrix product state
  #psi0 = randomMPS(sites, D_init)





  states_string = [2,1]


  for i in 1:Int(N/2)-2
    push!(states_string, 8) #add the lower state
    push!(states_string, 5) #add the upper state
  end

  push!(states_string, 2) #add the lower edge state
  push!(states_string, 3) #add the upper edge state
  
  psi_string = productMPS(sites, states_string)



  states_broken = [1,5,3,1]


  for i in 1:Int(N/2)-4
    push!(states_broken, 8) #add the lower state
    push!(states_broken, 8) #add the upper state
  end

  push!(states_broken, 2) 
  push!(states_broken, 2)
  push!(states_broken, 1) #add the lower edge state
  push!(states_broken, 5) #add the upper edge state

  psi_broken = productMPS(sites, states_broken)


  # Run the DMRG algorithm, returning energy and optimized MPS




  if g!=3.




    f = h5open("D3_ground_and_excited_"*string(N)*"_"*string(index_g-1)*".h5","r")


    psi_string = read(f,"psi_string",MPS)
    psi_broken = read(f,"psi_broken",MPS)

    close(f)

  end

  ff = h5open("D3_ground_and_excited_"*string(N)*"_"*string(index_g)*".h5","w")

  write(ff,"sites",sites)



  if N <= N_C

    energy, psi = dmrg(H, psi_string; nsweeps, observer = MyObserver,noise = noise, maxdim, cutoff)
    println("Starting second DMRG")
    energy_1,psi_1 = dmrg(H,[(max_val+1)*psi],psi_broken; nsweeps, observer = MyObserver_1,noise = noise, maxdim, cutoff)
    println("Second DMRG finished")
    write(ff,"psi_string",psi)
    write(ff,"psi_broken",psi_1)

  else


    energy, psi = dmrg(H, psi_broken; nsweeps, observer = MyObserver,noise = noise, maxdim, cutoff)
    println("Starting second DMRG")
    energy_1,psi_1 = dmrg(H,[(max_val+1)*psi],psi_string; nsweeps, observer = MyObserver_1,noise = noise, maxdim, cutoff)
    println("Second DMRG finished")
    write(ff,"psi_string",psi_1)
    write(ff,"psi_broken",psi)

  end

  close(ff)


  expect_values_e = zeros(Float64, N)
  expect_values_e_1 = zeros(Float64, N)



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

  for i in 2:Int(N/2)-1
    os = OpSum()
    os += 1,"E_u",2*i
    E_u = MPO(os, sites)
    expect_values_e_1[2*i] = real(inner(psi_1,Apply( E_u, psi_1)))
    os = OpSum()
    os += 1,"E_d",2*i-1
    E_d = MPO(os, sites)
    expect_values_e_1[2*i-1] = real(inner(psi_1,Apply( E_d, psi_1)))

  end

  expect_values_e_1[1] = real(inner(psi_1,Apply( E_dl, psi_1)))
  expect_values_e_1[2] = real(inner(psi_1,Apply( E_ul, psi_1)))
  expect_values_e_1[N-1] = real(inner(psi_1,Apply( E_dr, psi_1)))
  expect_values_e_1[N] = real(inner(psi_1,Apply( E_ur, psi_1)))



  res=string(N,"  ", g,"  ",D_max,"  ",energy,"  ",energy_1,"  ",expect_values_e,"  ",expect_values_e_1,"\n")

  open("results", "a") do io
      write(io, res)
  end



  psi = nothing
  psi_1 = nothing
  H = nothing





  







end



main(parse(Int64,ARGS[1]),parse(Float64,ARGS[2]),parse(Float64,ARGS[3]),parse(Int64,ARGS[4]))
