import sys
from bias_variance import (
    target_sin, 
    target_quad,
    monte_carlo_simulation,
    generate_learning_curve,
    plot_bias_variance,
    plot_learning_curve,
    print_comparison_table
)

def main():
    print("==================================================")
    print(" Bias-Variance Tradeoff & Generalization ")
    print("==================================================")
    
    # 1. เลือกฟังก์ชันเป้าหมาย
    print("\n[1] เลือกฟังก์ชันเป้าหมาย (Target Function):")
    print("  1. sin(pi * x)")
    print("  2. x^2")
    try:
        t_choice = int(input("เลือก (1 หรือ 2) [ค่าเริ่มต้น=1]: ") or "1")
    except ValueError:
        t_choice = 1
    
    if t_choice == 2:
        target_func = target_quad
        target_name = "f(x) = x^2"
    else:
        target_func = target_sin
        target_name = "f(x) = sin(pi * x)"

    # 2. จำนวนจุดข้อมูลฝึก
    try:
        n_samples = int(input("\n[2] ใส่จำนวนจุดข้อมูลฝึก (N) [ค่าเริ่มต้น=2]: ") or "2")
    except ValueError:
        n_samples = 2

    # 3. จำนวนรอบ Monte Carlo
    try:
        mc_iters = int(input("\n[3] ใส่จำนวนรอบ Monte Carlo Simulation [ค่าเริ่มต้น=2000]: ") or "2000")
    except ValueError:
        mc_iters = 2000

    # 4. ตั้งค่า Gaussian Noise
    add_noise_input = input("\n[4] เปิดใช้งาน Gaussian Noise หรือไม่? (y/n) [ค่าเริ่มต้น=n]: ").lower()
    add_noise = (add_noise_input == 'y')
    sigma = 0.0
    if add_noise:
        try:
            sigma = float(input("ใส่ค่า Sigma (ความแปรปรวนของ Noise) [ค่าเริ่มต้น=0.1]: ") or "0.1")
        except ValueError:
            sigma = 0.1

    print("\nกำลังประมวลผล Monte Carlo Simulation เพื่อเปรียบเทียบทุกโมเดล...")
    
    models_to_test = {
        'Constant Model': 'constant',
        'Linear Model': 'linear',
        'Linear Through Origin': 'linear_origin'
    }
    
    results_all = {}
    
    # รัน Monte Carlo สำหรับทุกโมเดลเพื่อสร้างตารางเปรียบเทียบ
    for m_name, m_type in models_to_test.items():
        res = monte_carlo_simulation(
            target_func=target_func,
            model_type=m_type,
            n_samples=n_samples,
            mc_iterations=mc_iters,
            add_noise=add_noise,
            sigma=sigma
        )
        results_all[m_name] = res
    
    # พิมพ์ตารางเปรียบเทียบ (Bias, Variance, E_out)
    print_comparison_table(results_all)
    
    # 5. เลือกโมเดลที่ต้องการดูกราฟ
    print("\n[5] เลือกโมเดลเพื่อดูกราฟ Bias-Variance และ Learning Curve:")
    print("  1. Constant Model (h(x) = b)")
    print("  2. Linear Model (h(x) = ax + b)")
    print("  3. Linear Through Origin (h(x) = ax)")
    try:
        m_choice = int(input("เลือก (1, 2, หรือ 3) [ค่าเริ่มต้น=2]: ") or "2")
    except ValueError:
        m_choice = 2
        
    if m_choice == 1:
        sel_name, sel_type = 'Constant Model', 'constant'
    elif m_choice == 3:
        sel_name, sel_type = 'Linear Through Origin', 'linear_origin'
    else:
        sel_name, sel_type = 'Linear Model', 'linear'
        
    print(f"\nแสดงผลกราฟสำหรับ {sel_name} ...")
    
    # แสดงกราฟ Bias-Variance
    plot_bias_variance(results_all[sel_name], f"{sel_name} on {target_name} (N={n_samples})")
    
    # 6. จำนวนจุดสำหรับ Learning Curve
    try:
        max_n = int(input("\n[6] ใส่ค่า N สูงสุดสำหรับสร้าง Learning Curve [ค่าเริ่มต้น=20]: ") or "20")
    except ValueError:
        max_n = 20
        
    print("\nกำลังคำนวณ Learning Curve...")
    n_vals, e_ins, e_outs = generate_learning_curve(
        target_func=target_func,
        model_type=sel_type,
        max_n=max_n,
        mc_iterations=500,  # ใช้ 500 รอบเพื่อความรวดเร็วในการพลอต
        add_noise=add_noise,
        sigma=sigma
    )
    
    # แสดงกราฟ Learning Curve
    plot_learning_curve(n_vals, e_ins, e_outs, sel_name)
    
    print("\nจบการทำงาน!")

if __name__ == "__main__":
    # การทำงานแบบ Interactive Script
    main()
