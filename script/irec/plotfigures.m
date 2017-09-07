x = 0:0.001:1;
k = 1.3;
k1 = 8;
y0 = k1 /(k1-k) * (1 - k * x);
y0(y0>1) = 1;
y0(y0<0) = 0;
y1 = x<=1/k;
plot(x,y1,'-','LineWidth',3); hold on
plot(x,y0, '-.','LineWidth',3); hold on
legend('hard weighting', 'soft weighting')
plot([1/k1, 1/k1], [0,1], '--k')
text(1/k1+0.007,0.08, '$\frac{1}{k''}$', 'interpreter','latex', 'fontsize',16)
text(1/k+0.007,0.09, '$\frac{1}{k}$', 'interpreter','latex', 'fontsize',16)
ylim([0,1.3])
xlabel('average loss $\bar{\ell}_i$', 'interpreter','latex');
ylabel('$\theta_i$','interpreter','latex')
ApplyFigTemplate(gcf,gca)
print -dpdf abc.pdf
