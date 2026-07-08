import numpy as np
import matplotlib.pyplot as plt
from bias_variance import (
    target_sin, 
    target_quad,
    generate_learning_curve
)

def plot_combined_learning_curve(target_func, target_name, max_n, mc_iterations, add_noise=False, sigma=0.1):
    """
    พลอตเส้นโค้งการเรียนรู้เปรียบเทียบ 3 แบบจำลอง (Constant, Linear, Linear Origin) 
    บนกราฟเดียวกัน
    """
    print(f"\nกำลังคำนวณเส้นโค้งการเรียนรู้สำหรับฟังก์ชันเป้าหมาย: {target_name}")
    if add_noise:
        print(f"มีการใส่สัญญาณรบกวน (Gaussian Noise) ระดับ sigma = {sigma}")
    else:
        print("ไม่มีสัญญาณรบกวน (No Noise)")
    
    models = {
        'Constant (h(x)=b)': 'constant',
        'Linear (h(x)=ax+b)': 'linear',
        'Linear Origin (h(x)=ax)': 'linear_origin'
    }
    
    plt.figure(figsize=(15, 6))
    
    colors = {
        'Constant (h(x)=b)': 'blue',
        'Linear (h(x)=ax+b)': 'red',
        'Linear Origin (h(x)=ax)': 'green'
    }
    
    # พลอต E_in (Training Error)
    plt.subplot(1, 2, 1)
    for m_name, m_type in models.items():
        n_vals, e_ins, e_outs = generate_learning_curve(
            target_func=target_func,
            model_type=m_type,
            max_n=max_n,
            mc_iterations=mc_iterations,
            add_noise=add_noise,
            sigma=sigma
        )
        plt.plot(n_vals, e_ins, marker='o', label=m_name, color=colors[m_name])
        
        # จัดเก็บ e_outs เพื่อไปพลอตกราฟด้านขวา
        models[m_name] = {'n_vals': n_vals, 'e_outs': e_outs}
        
    plt.title(f'Training Error ($E_{{in}}$) - {target_name}')
    plt.xlabel('Number of Training Examples (N)')
    plt.ylabel('Expected $E_{{in}}$')
    plt.legend()
    plt.grid(True)
    
    # พลอต E_out (Expected Out-of-Sample Error)
    plt.subplot(1, 2, 2)
    for m_name, data in models.items():
        if isinstance(data, dict): # เช็คว่าประมวลผลแล้ว
            plt.plot(data['n_vals'], data['e_outs'], marker='o', label=m_name, color=colors[m_name])
            
    plt.title(f'Expected Error ($E_{{out}}$) - {target_name}')
    plt.xlabel('Number of Training Examples (N)')
    plt.ylabel('Expected $E_{{out}}$')
    
    # ตั้งค่า ylim เล็กน้อยเพื่อไม่ให้ค่าที่พุ่งสูงตอน N น้อยๆ ทำให้ดูกราฟยาก
    # (โดยเฉพาะ Linear Model ตอน N=2 ที่ E_out มักจะพุ่งสูงมาก)
    plt.ylim(0, 3.0 if target_name == "sin(pi*x)" else 2.0)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("==================================================")
    print(" Learning Curve Comparison (เส้นโค้งการเรียนรู้) ")
    print("==================================================")
    
    # 1. ทดลองบนฟังก์ชัน sin(pi*x) โดยไม่มี noise
    plot_combined_learning_curve(target_sin, "sin(pi*x)", max_n=15, mc_iterations=500, add_noise=False)
    
    # 2. ทดลองบนฟังก์ชัน sin(pi*x) โดยใส่สัญญาณรบกวน (Gaussian noise, sigma=0.5)
    plot_combined_learning_curve(target_sin, "sin(pi*x) with Noise (sigma=0.5)", max_n=15, mc_iterations=500, add_noise=True, sigma=0.5)
    
    # 3. ทดลองบนฟังก์ชัน x^2 โดยไม่มี noise
    plot_combined_learning_curve(target_quad, "x^2", max_n=15, mc_iterations=500, add_noise=False)
    
    # 4. ทดลองบนฟังก์ชัน x^2 โดยใส่สัญญาณรบกวน (Gaussian noise, sigma=0.5)
    plot_combined_learning_curve(target_quad, "x^2 with Noise (sigma=0.5)", max_n=15, mc_iterations=500, add_noise=True, sigma=0.5)
