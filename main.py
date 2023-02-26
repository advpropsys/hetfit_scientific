import streamlit as st

from nets.envs import SCI


st.set_page_config(
        page_title="HET_sci",
        menu_items={
            'About':'https://advpropsys.github.io'
        }
)

st.title('HETfit_scientific')
st.markdown("#### Imagine a package which was engineered primarly for data driven plasma physics devices design, mainly low power hall effect thrusters, yup that's it"
            "\n### :orange[Don't be scared away though, it has much simpler interface than anything you ever used for such designs]")
st.markdown('### Main concepts:')
st.markdown( "- Each observational/design session is called an **environment**, for now it can be either RCI or SCI (Real or scaled interface)"
            "\n In this overview we will only touch SCI, since RCI is using PINNs which are different topic"
            "\n- You specify most of the run parameters on this object init, :orange[**including generation of new samples**] via GAN"
            "\n- You may want to generate new features, do it !"
            "\n- Want to select best features for more effctive work? Done!"
            "\n- Compile environment with your model of choice, can be ***any*** torch model or sklearn one"
            "\n- Train !"
            "\n- Plot, inference, save, export to jit/onnx, measure performance - **they all are one liners** "
            )
st.markdown('### tl;dr \n- Create environment'
            '\n```run = SCI(*args,**kwargs)```'
            '\n - Generate features ```run.feature_gen()``` '
            '\n - Select features ```run.feature_importance()```'
            '\n - Compile env ```run.compile()```'
            '\n - Train model in env ```run.train()```'
            '\n - Inference, plot, performance, ex. ```run.plot3d()```'
            '\n #### And yes, it all will work even without any additional arguments from user besides column indexes'
            )
st.write('Comparison with *arXiv:2206.04440v3*')
col1, col2 = st.columns(2)
col1.metric('Geometry accuracy on domain',value='83%',delta='15%')
col2.metric('$d \mapsto h$ prediction',value='98%',delta='14%')

st.header('Example:')

st.markdown('Remeber indexes and column names on this example: $P$ - 1, $d$ - 3, $h$ - 3, $m_a$ - 6,$T$ - 7')
st.code('run = SCI(*args,**kwargs)')

run = SCI()
st.code('run.feature_gen()')
run.feature_gen()
st.write('New features: (index-0:22 original samples, else is GAN generated)',run.df.iloc[1:,9:].astype(float))
st.write('Most of real dataset is from *doi:0.2514/1.B37424*, hence the results mostly agree with it in specific')
st.code('run.feature_importance(run.df.iloc[1:,1:7].astype(float),run.df.iloc[1:,7]) # Clear and easy example')

st.write(run.feature_importance(run.df.iloc[1:,1:6].astype(float),run.df.iloc[1:,6]))
st.markdown(' As we can see only $h$ and $d$ passed for $m_a$ model, not only that linear dependacy was proven experimantally, but now we got this from data driven source')
st.code('run.compile(idx=(1,3,7))')
run.compile(idx=(1,3,7))
st.code('run.train(epochs=10)')
if st.button('Start Training',icon='⏳',use_container_width=True):
    run.train(epochs=10)
st.code('run.plot3d()')
st.write(run.plot3d())
st.code('run.performance()')
st.write(run.performance())

st.write('Try it out yourself! Select a column from 1 to 10')
numcol,button = st.columns(2)

number = numcol.number_input('Here',min_value=1, max_value=10, step=1)

if number:
    if button.button('Compile And Train💅'):
        st.code(f'run.compile(idx=(1,3,{number}))')
        run.compile(idx=(1,3,number))
        st.code('run.train(epochs=10)')
        run.train(epochs=10)
        st.code('run.plot3d()')
        st.write(run.plot3d())



st.markdown('In this intro we covered simplest userflow while using HETFit package, resulted data can be used to leverage PINN and analytical models of Hall effect thrusters'
            '\n #### :orange[To cite please contact author on https://github.com/advpropsys]')