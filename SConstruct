# SConscript('main.scons', variant_dir='build', duplicate=0)
import distutils.sysconfig, os
import platform
if platform.system() == "Windows":
    # Update these values according to your installation directories
    BOOST_ROOT=os.environ.get('BOOST_ROOT') # 64-bit binaries (boost_1.62.0) downloaded from https://boost.teeks99.com
    BOOST_LIBDIR=BOOST_ROOT + r"\lib64-msvc-14.0"
    # Make sure both python and boost binaries are either x64 or x86 and accordingly update path to opencv binaries
    OPENCV_DIR=os.environ.get('OPENCV_PATH')
    OPENCV_VER="2410"
def TOOL_BOOST_DISTUTILS(env):
    """Add stuff needed to build Python extensions with boost.  """
    vars = distutils.sysconfig.get_config_vars('CC', 'CXX', 'OPT', 'BASECFLAGS', 'CCSHARED', 'LDSHARED', 'SO')
    for i in range(len(vars)):
        if vars[i] is None:
            vars[i] = ""
    (cc, cxx, opt, basecflags, ccshared, ldshared, so_ext) = vars
    if platform.system() == "Windows":
        env.AppendUnique(CPPPATH=[distutils.sysconfig.get_python_inc(),BOOST_ROOT, \
        distutils.sysconfig.PREFIX + r'\Lib\site-packages\numpy\core\include',
        OPENCV_DIR + r'\build\include'])
        env.AppendUnique(CXXFLAGS=Split("/Zm800 -nologo /EHsc /DBOOST_PYTHON_DYNAMIC_LIB /Z7 /Od /Ob0 /EHsc /GR /MD  /wd4675 /Zc:forScope /Zc:wchar_t"))
        env.AppendUnique(LIBPATH=[BOOST_LIBDIR, distutils.sysconfig.PREFIX+"/libs", OPENCV_DIR + r'\build\x64\vc12\lib'])
        env.AppendUnique(LIBS=["boost_python-vc140-mt-gd-1_62",'opencv_features2d' + OPENCV_VER, 'opencv_core' + OPENCV_VER, \
        'opencv_highgui' + OPENCV_VER, 'opencv_imgproc' + OPENCV_VER])
        
    env['SHLIBPREFIX']=""   #gets rid of lib prefix
    env['SHLIBSUFFIX']=so_ext
    
Default('.')
if platform.system() == "Windows":
    env=Environment(tools=['default', TOOL_BOOST_DISTUTILS], \
    MSVC_USE_SCRIPT=r'C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat')
env.VariantDir('build', 'jni', duplicate=0)
lib = env.SharedLibrary('active_contour', Glob('build/*.cpp'))
env.Install("src/", lib)