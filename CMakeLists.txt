cmake_minimum_required(VERSION 3.16)
project(omnetpp_suite NONE)

include(ProcessorCount)
ProcessorCount(_proc_count)
if (NOT _proc_count OR _proc_count LESS 1)
  set(_proc_count 1)
endif()
set(MAKE_JOBS "${_proc_count}" CACHE STRING "Parallel jobs for custom make targets")

set(OMNETPP_ROOT "${CMAKE_SOURCE_DIR}/omnetpp-6.3.0")
set(INET_ROOT "${CMAKE_SOURCE_DIR}/inet")
set(SIMU5G_ROOT "${CMAKE_SOURCE_DIR}/Simu5G")
set(SIMU5G_GYM_ROOT "${CMAKE_SOURCE_DIR}/Simu5G-Gym")

set(ENV_OMNETPP "source ${OMNETPP_ROOT}/setenv -q")
set(ENV_OMNETPP_INET "${ENV_OMNETPP} && source ${INET_ROOT}/setenv -q")

function(add_make_target name env_cmd workdir)
  string(JOIN " " make_args ${ARGN})
  if (make_args STREQUAL "")
    set(make_cmd "make -j${MAKE_JOBS}")
  else()
    set(make_cmd "make -j${MAKE_JOBS} ${make_args}")
  endif()
  set(full_cmd "${env_cmd} && cd ${workdir} && ${make_cmd}")
  # Escape && so the outer shell doesn't split the bash -lc argument.
  string(REPLACE "&&" "\\&\\&" full_cmd "${full_cmd}")
  add_custom_target(${name}
    COMMAND bash -lc "${full_cmd}"
    USES_TERMINAL
  )
endfunction()

# OMNeT++
add_make_target(omnetpp "${ENV_OMNETPP}" "${OMNETPP_ROOT}")
add_make_target(omnetpp_release "${ENV_OMNETPP}" "${OMNETPP_ROOT}" MODE=release)
add_make_target(omnetpp_debug "${ENV_OMNETPP}" "${OMNETPP_ROOT}" MODE=debug)
add_make_target(omnetpp_clean "${ENV_OMNETPP}" "${OMNETPP_ROOT}" clean)
add_make_target(omnetpp_cleanall "${ENV_OMNETPP}" "${OMNETPP_ROOT}" cleanall)

# INET
add_make_target(inet_makefiles "${ENV_OMNETPP_INET}" "${INET_ROOT}" makefiles)
add_make_target(inet "${ENV_OMNETPP_INET}" "${INET_ROOT}")
add_make_target(inet_debug "${ENV_OMNETPP_INET}" "${INET_ROOT}" MODE=debug)
add_make_target(inet_clean "${ENV_OMNETPP_INET}" "${INET_ROOT}" clean)
add_make_target(inet_cleanall "${ENV_OMNETPP_INET}" "${INET_ROOT}" cleanall)
add_dependencies(inet omnetpp_release)
add_dependencies(inet_debug omnetpp_debug)

# Simu5G
add_make_target(simu5g_makefiles "${ENV_OMNETPP_INET}" "${SIMU5G_ROOT}" makefiles)
add_make_target(simu5g "${ENV_OMNETPP_INET}" "${SIMU5G_ROOT}")
add_make_target(simu5g_debug "${ENV_OMNETPP_INET}" "${SIMU5G_ROOT}" MODE=debug)
add_make_target(simu5g_tests "${ENV_OMNETPP_INET}" "${SIMU5G_ROOT}" tests)
add_make_target(simu5g_clean "${ENV_OMNETPP_INET}" "${SIMU5G_ROOT}" clean)
add_make_target(simu5g_cleanall "${ENV_OMNETPP_INET}" "${SIMU5G_ROOT}" cleanall)
add_dependencies(simu5g inet)
add_dependencies(simu5g_debug inet_debug)

# Simu5G-Gym
add_make_target(simu5g_gym_makefiles "${ENV_OMNETPP_INET}" "${SIMU5G_GYM_ROOT}" makefiles)
add_make_target(simu5g_gym "${ENV_OMNETPP_INET}" "${SIMU5G_GYM_ROOT}")
add_make_target(simu5g_gym_debug "${ENV_OMNETPP_INET}" "${SIMU5G_GYM_ROOT}" MODE=debug)
add_make_target(simu5g_gym_clean "${ENV_OMNETPP_INET}" "${SIMU5G_GYM_ROOT}" clean)
add_make_target(simu5g_gym_cleanall "${ENV_OMNETPP_INET}" "${SIMU5G_GYM_ROOT}" cleanall)
add_dependencies(simu5g_gym simu5g)
add_dependencies(simu5g_gym_debug simu5g_debug)

add_custom_target(all_release)
add_dependencies(all_release omnetpp_release inet simu5g simu5g_gym)

add_custom_target(all_debug)
add_dependencies(all_debug omnetpp_debug inet_debug simu5g_debug simu5g_gym_debug)
