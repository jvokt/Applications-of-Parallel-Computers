--
-- Timing parameter file: time different numbers of points
--

function gaussian(x)
  local stddev = 0.1
  local mean = 0.5
  local var2 = stddev*stddev*2
  local term = x-mean
  return stddev * math.exp(-term*term/var2)/math.sqrt(math.pi*var2)
end


--
-- Helper: Print only on processor 0
--
function print0(...)
  if rank == 0 then print(...) end
end


--
-- Time sims with 10K-100K points in steps of 10K
--
print0("n,t")
for ns = 1,10 do
  t = run {
      u0 = gaussian,   -- Initial conditions
      n = ns*10000,    -- Number of primary mesh points
      nsteps = 4000,   -- Number of time steps
      fstep = 10,      -- Time between frames
      c = 0.34029,     -- Speed of sounds
      verbose = 0      -- Suppress verbose output
  }
  print0((ns*10000) .. "," .. t) 
end
