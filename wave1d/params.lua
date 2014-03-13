--
-- Example parameter file: wave equation starting from a Gaussian bump
--

function gaussian(x)
  local stddev = 0.1
  local mean = 0.5
  local var2 = stddev*stddev*2
  local term = x-mean
  return stddev * math.exp(-term*term/var2)/math.sqrt(math.pi*var2)
end

run {
    fname = "u_plot.txt",   -- Output file
    u0 = gaussian,          -- Initial conditions
    n = 1000,               -- Number of primary mesh points
    nsteps = 4000,          -- Number of time steps
    fstep = 10,             -- Time steps between frames
    c = 0.34029,            -- Speed of sound
}
